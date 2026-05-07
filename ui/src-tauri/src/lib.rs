use std::process::Command;
use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandChild;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let sidecar_command = app
                .shell()
                .sidecar("api-server")
                .expect("Failed to create sidecar command");

            let (mut rx, child) = sidecar_command
                .spawn()
                .expect("Failed to start backend");
            let backend_pid = child.pid();

            // Save child handle so we can kill it on window close
            app.manage(BackendProcess {
                child: Mutex::new(Some(child)),
                pid: backend_pid,
            });

            // Listen for sidecar events in background to capture errors
            tauri::async_runtime::spawn(async move {
                use tauri_plugin_shell::process::CommandEvent;
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            eprintln!("[api-server] {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Stderr(line) => {
                            eprintln!("[api-server:err] {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Terminated(status) => {
                            eprintln!("[api-server] exited with {:?}", status);
                        }
                        CommandEvent::Error(err) => {
                            eprintln!("[api-server:error] {}", err);
                        }
                        _ => {}
                    }
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(
                event,
                tauri::WindowEvent::CloseRequested { .. } | tauri::WindowEvent::Destroyed
            ) {
                stop_backend(&window.app_handle());
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/// Holds the backend sidecar process handle for cleanup on window close
struct BackendProcess {
    child: Mutex<Option<CommandChild>>,
    pid: u32,
}

fn stop_backend(app: &tauri::AppHandle) {
    if let Some(state) = app.try_state::<BackendProcess>() {
        if let Ok(mut guard) = state.child.lock() {
            if let Some(child) = guard.take() {
                kill_process_tree(state.pid);
                let _ = child.kill();
            }
        }
    }
}

fn kill_process_tree(pid: u32) {
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .creation_flags(CREATE_NO_WINDOW)
            .status();
    }

    #[cfg(not(windows))]
    {
        let _ = Command::new("kill").args(["-TERM", &pid.to_string()]).status();
    }
}
