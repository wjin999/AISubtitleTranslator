import re
import os

def parse_srt_file(filepath):
    """
    解析 SRT 文件，返回一个列表，每个元素对应一条字幕：
    [
      {
        'index': '1',
        'time': '00:00:00,000 --> 00:00:05,233',
        'text_lines': ['I coached Overwatch for eight years,', 'and I recently had a student...']
      },
      {
        'index': '2',
        'time': '00:00:05,234 --> 00:00:10,000',
        'text_lines': ['...']
      },
      ...
    ]
    """
    subtitles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    temp_block = []
    for line in lines:
        # 每段字幕之间通常有一个空行，当检测到空行，说明前一段结束
        if line.strip() == "":
            if temp_block:
                subtitles.append(process_block(temp_block))
                temp_block = []
        else:
            temp_block.append(line)

    # 文件末尾可能没有空行，需要再处理一次
    if temp_block:
        subtitles.append(process_block(temp_block))

    return subtitles

def process_block(block_lines):
    """
    处理单个字幕块（列表），将其解析成 {index, time, text_lines} 字典
    SRT 格式通常:
        1
        00:00:00,000 --> 00:00:05,233
        文本1
        文本2
        （空行）
    """
    # block_lines[0] => index
    # block_lines[1] => time
    # block_lines[2..] => text lines
    subtitle_dict = {
        'index': block_lines[0].strip(),
        'time': block_lines[1].strip(),
        'text_lines': block_lines[2:]
    }
    return subtitle_dict

def is_sentence_complete(text_block):
    """
    判断某个字幕块是否以句号、问号、感叹号结尾。
    这里我们以该字幕块的最后一行文本是否以这些标点结尾进行简单判断。
    如果需要更严格/复杂的判断逻辑，可以自己修改。
    """
    if not text_block['text_lines']:
        return True  # 如果连文本都没有，默认认为这条是“完整”（或可以按需求返回 False）
    last_line = text_block['text_lines'][-1].rstrip()  # 取最后一行文本
    # 简单判断是否以 '.', '?', '!' 结尾
    return bool(re.search(r'[.!?]$', last_line))

def split_srt_file(input_filepath, chunk_size=200, output_dir='.'):
    """
    根据给定的 chunk_size（默认 200 条字幕），分割原始 SRT 文件。
    如果最后一条字幕不是完整句子，则继续往后找，直到找到完整句子。
    分割后输出多个文件：1.srt, 2.srt, 3.srt, ...
    其中：
      - 第一个文件包含第 1～(200+x) 条字幕
      - 第二个文件包含第 (201+x)～(400+y) 条字幕
      - ...
    """
    # 1. 解析原始 SRT 文件
    subtitles = parse_srt_file(input_filepath)
    total = len(subtitles)
    print(f"共解析到 {total} 条字幕。")

    start_index = 0  # 当前切分段的起始下标
    file_count = 0   # 输出文件计数

    # 2. 按 chunk_size 循环切分
    while start_index < total:
        file_count += 1
        # 预定的结束下标（不含）
        end_index = start_index + chunk_size  
        if end_index >= total:
            # 如果超出总数，则直接取到末尾
            end_index = total
        else:
            # 如果没到末尾，就检查结尾字幕是否完整
            while end_index < total and not is_sentence_complete(subtitles[end_index - 1]):
                end_index += 1
        
        # 切片当前段的字幕
        current_slice = subtitles[start_index:end_index]

        # 写入新的 SRT 文件
        output_filename = os.path.join(output_dir, f"{file_count}.srt")
        write_srt_file(output_filename, current_slice)
        print(f"输出文件：{output_filename}，包含第 {current_slice[0]['index']}-{current_slice[-1]['index']} 条字幕。")

        # 更新下一段起始下标
        start_index = end_index

    print("分割完成！")

def write_srt_file(filepath, subtitle_blocks):
    """
    将给定的字幕块列表写入新的 SRT 文件，保持原始的编号、时间轴、文本。
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, block in enumerate(subtitle_blocks):
            f.write(block['index'] + '\n')
            f.write(block['time'] + '\n')
            for line in block['text_lines']:
                f.write(line + '\n')
            f.write('\n')  # 每条字幕后空行

if __name__ == "__main__":
    # 示例用法
    input_srt = r""
    output_directory = os.path.dirname(input_srt)
    split_srt_file(input_srt, chunk_size=200, output_dir=output_directory)
