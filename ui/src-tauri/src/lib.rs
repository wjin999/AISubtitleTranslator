use tauri::Manager;
use tauri_plugin_shell::ShellExt;
// 引入 Windows 专属的命令行扩展特性
use std::os::windows::process::CommandExt;

// 这是 Windows 系统底层的魔法常量，表示“隐身运行，不要弹窗”
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let sidecar_command = app.shell().sidecar("api-server").expect("Failed to create sidecar command");
            
            tauri::async_runtime::spawn(async move {
                let (_rx, _child) = sidecar_command.spawn().expect("Failed to start backend");
            });

            Ok(())
        })
        .on_window_event(|_window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                // 绝杀 1：带上“隐身斗篷”的 taskkill
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/T", "/IM", "api-server.exe"])
                    .creation_flags(CREATE_NO_WINDOW) // <--- 关键：隐藏黑框
                    .status();
                
                // 绝杀 2：同样隐身
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/T", "/IM", "api-server-x86_64-pc-windows-msvc.exe"])
                    .creation_flags(CREATE_NO_WINDOW) // <--- 关键：隐藏黑框
                    .status();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}