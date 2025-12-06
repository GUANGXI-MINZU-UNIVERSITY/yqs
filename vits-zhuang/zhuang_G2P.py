import os
import re
from functools import lru_cache
from tqdm import tqdm

# ... initials、finals等配置略 ...
initials = {
    'b': 'p', 'mb': 'b', 'm': 'm', 'f': 'f', 'v': 'v',
    'd': 't', 'nd': 'd', 'n': 'n', 's': 'θ', 'l': 'l',
    'g': 'k', 'gv': 'kʷ', 'ng': 'ŋ', 'h': 'h', 'r': 'ɣ',
    'c': 'ɕ', 'y': 'j', 'ny': 'ȵ', 'ngv': 'ŋʷ',
    'by': 'pʲ', 'gy': 'kʲ', 'my': 'mʲ'
}
finals = {
    'a': 'aː', 'e': 'eː', 'i': 'iː', 'o': 'oː', 'u': 'uː', 'w': 'ɯː',
    'ai': 'aːi', 'ae': 'ai', 'ei': 'ei', 'oi': 'oːi', 'ui': 'uːi	', 'wi': 'ɯːi',
    'au': 'aːu', 'aeu': 'au', 'eu': 'eːu', 'iu': 'iːu', 'ou': 'ou',
    'aw': 'aɯ',
    'am': 'aːm', 'aem': 'am', 'em': 'eːm', 'iem': 'iːm', 'im': 'im', 'om': 'oːm', 'oem': 'om', 'uem': 'uːm', 'um': 'um',
    'an': 'aːn', 'aen': 'an', 'en': 'eːn', 'ien': 'iːn', 'in': 'in', 'on': 'oːn', 'oen': 'on', 'uen': 'uːn', 'un': 'un',
    'wen': 'ɯːn', 'wn': 'ɯn',
    'ang': 'aːŋ', 'aeng': 'aŋ', 'eng': 'eːŋ', 'ieng': 'iːŋ', 'ing': 'iŋ', 'ong': 'oːŋ', 'oeng': 'oŋ', 'ueng': 'uːŋ',
    'ung': 'uŋ', 'wng': 'ɯn',
    'ap': 'aːp̚', 'aep': 'ap̚', 'ep': 'eːp̚', 'iep': 'iːp̚', 'ip': 'ip̚', 'op': 'oːp̚', 'oep': 'op̚', 'uep': 'uːp̚',
    'up': 'up̚',
    'at': 'aːt̚', 'aet': 'at̚', 'et': 'eːt̚', 'iet': 'iːt̚', 'it': 'it̚', 'ot': 'oːt̚', 'oet': 'ot̚', 'uet': 'uːt̚',
    'ut': 'ut̚', 'wet': 'ɯːt̚', 'wt': 'ɯt̚',
    'ak': 'aːk̚', 'aek': 'ak̚', 'ek': 'eːk̚', 'iek': 'iːk̚', 'ik': 'ik̚', 'ok': 'oːk̚', 'oek': 'ok̚', 'uek': 'uːk̚',
    'uk': 'uk̚', 'wk': 'ɯk̚',
    'ab': 'aːp̚', 'aeb': 'ap̚', 'eb': 'eːp̚', 'ieb': 'iːp̚', 'ib': 'ip̚', 'ob': 'oːp̚', 'oeb': 'op̚', 'ueb': 'uːp̚',
    'ub': 'up̚',
    'ad': 'aːt̚', 'aed': 'at̚', 'ed': 'eːt̚', 'ied': 'iːt̚', 'id': 'it̚', 'od': 'oːt̚', 'oed': 'ot̚', 'ued': 'uːt̚',
    'ud': 'ut̚', 'wed': 'ɯːt̚', 'wd': 'ɯt̚',
    'ag': 'aːk̚', 'aeg': 'ak̚', 'eg': 'eːk̚', 'ieg': 'iːk̚', 'ig': 'ik̚', 'og': 'oːk̚', 'oeg': 'ok̚', 'ueg': 'uːk̚',
    'ug': 'uk̚', 'wg': 'ɯk̚'
}
ipa_tone_symbols = {
    '1': '˨', '2': '˧', '3': '˦', '4': '˥', '5': '˦˨', '6': '˧˥', '7': '˥˧', '8': '˧˩'
}
stop_tone_endings_ipa = {'p': '˥˧', 't': '˥˧', 'k': '˥˧', 'b': '˧˩', 'd': '˧˩', 'g': '˧˩'}

tone_symbols_to_numbers = {'z': '2', 'j': '3', 'x': '4', 'q': '5', 'h': '6'}
stop_tone_endings = {'p': '7', 't': '7', 'k': '7', 'b': '8', 'd': '8', 'g': '8'}
tone_marker = '#'

number_to_pinyin_tone_letter = {
    '1': '',    # 1声调不标
    '2': 'z',
    '3': 'j',
    '4': 'x',
    '5': 'q',
    '6': 'h',
    '7': '',    # 7和8为入声，不标拼音调字母
    '8': ''
}

sorted_initials = sorted(initials.keys(), key=lambda x: -len(x))
sorted_finals = sorted(finals.keys(), key=lambda x: -len(x))

failed_words = set()
tilde_regex = re.compile(r'(?<=.)~|~(?=.)')
split_regex = re.compile(r'\s+')

# ===================== 基础处理 =====================
def split_words(sentence):
    words = []
    retain_symbols = {'%', '~', '.', '+', '-', '*', '/'}
    for raw_word in sentence.split():
        prefix, core, suffix = '', raw_word, ''
        while core and not (core[0].isalnum() or core[0] in retain_symbols):
            prefix += core[0]
            core = core[1:]
        while core and not (core[-1].isalnum() or core[-1] in retain_symbols):
            suffix = core[-1] + suffix
            core = core[:-1]
        words.append((prefix, core.lower(), suffix))
    return words

def is_zhuang_word(word):
    if re.match(r'^[\d\W_]+$', word):
        return False
    return any(c.isalpha() or c in tone_symbols_to_numbers for c in word)

def parse_syllable(word):
    initial = ''
    for init in sorted(initials.keys(), key=len, reverse=True):
        if word.startswith(init):
            initial = init
            remaining = word[len(init):]
            break
    else:
        remaining = word

    final = ''
    for fin in sorted(finals.keys(), key=len, reverse=True):
        if remaining.startswith(fin):
            final = fin
            after_final = remaining[len(fin):]
            if after_final and after_final[0] in 'zjxqh':
                tone = after_final[0]
                leftover = after_final[1:]
            elif fin and fin[-1] in stop_tone_endings:
                tone = fin[-1]
                leftover = after_final
            else:
                tone = '1'
                leftover = after_final
            return initial, fin, tone, leftover
    return '', remaining, '1', ''

def process_word(word):
    parts = []
    max_loops = len(word) * 2 + 10
    loop_count = 0
    while word and loop_count < max_loops:
        loop_count += 1
        if word[0].isalpha():
            result = parse_syllable(word)
            initial, final, tone, remaining = result
            if initial or final:
                parts.extend([
                    initial if initial else '',
                    final if final else '',
                    tone + tone_marker if tone else ''
                ])
            else:
                parts.append(word)
                break
            word = remaining
        else:
            char = word[0]
            word = word[1:]
            if char == '~':
                parts.append(' ~ ')
            else:
                parts.append(char)
    return [x for x in parts if x]

# ===================== 各种模式 =====================
def convert_zhuang(sentence, mode):
    """
    mode:
        'ipa_ipa'    国际音标+IPA声调
        'ipa_num'    国际音标+数字#号
        'py_ipa'     拼音+IPA声调
        'py_num'     拼音+数字#号
        'py_py'      拼音+拼音声调（只拆分，不输出声调数字/符号，1声调不标）
        'py_pyhash'  拼音+拼音声调+#号（1声调不标）
        'ipa_with_tone' 壮文直接转IPA，原有空格保留，不添加新空格
    """
    if mode == 'py_py':
        return zhuang_mode_py_py(sentence)
    if mode == 'py_pyhash':
        return zhuang_mode_py_pyhash(sentence)
    if mode == 'ipa_ipa':
        return zhuang_mode_ipa_ipa(sentence)
    if mode == 'ipa_num':
        return zhuang_mode_ipa_num(sentence)
    if mode == 'py_ipa':
        return zhuang_mode_py_ipa(sentence)
    if mode == 'py_num':
        return zhuang_mode_py_num(sentence)
    if mode == 'ipa_with_tone':
        return zhuang_to_ipa_with_tone(sentence)
    raise ValueError(f"Unknown mode: {mode}")

def zhuang_mode_py_py(sentence):
    word_strings = []
    for prefix, core, suffix in split_words(sentence):
        word_parts = []
        if prefix:
            word_parts.append(prefix)
        if is_zhuang_word(core):
            processed = process_word(core)
            for item in processed:
                if tone_marker in item:
                    tone_char = item.replace(tone_marker, '')
                    if tone_char in tone_symbols_to_numbers:
                        num = tone_symbols_to_numbers[tone_char]
                    elif tone_char in stop_tone_endings:
                        num = stop_tone_endings[tone_char]
                    else:
                        num = '1'
                    py_letter = number_to_pinyin_tone_letter.get(num, '')
                    if py_letter:
                        word_parts.append(py_letter)
                else:
                    word_parts.append(item)
        else:
            word_parts.append(core)
        if suffix:
            word_parts.append(suffix)
        word_str = ' '.join(filter(None, word_parts))
        word_strings.append(word_str)
    return ' '.join(word_strings)

def zhuang_mode_py_pyhash(sentence):
    word_strings = []
    for prefix, core, suffix in split_words(sentence):
        word_parts = []
        if prefix:
            word_parts.append(prefix)
        if is_zhuang_word(core):
            processed = process_word(core)
            for item in processed:
                if tone_marker in item:
                    tone_char = item.replace(tone_marker, '')
                    if tone_char in tone_symbols_to_numbers:
                        num = tone_symbols_to_numbers[tone_char]
                    elif tone_char in stop_tone_endings:
                        num = stop_tone_endings[tone_char]
                    else:
                        num = '1'
                    py_letter = number_to_pinyin_tone_letter.get(num, '')
                    if py_letter:
                        word_parts.append(f"{py_letter}{tone_marker}")
                    elif num in ['7', '8']:
                        pass
                else:
                    word_parts.append(item)
        else:
            word_parts.append(core)
        if suffix:
            word_parts.append(suffix)
        word_str = ' '.join(filter(None, word_parts))
        word_strings.append(word_str)
    return ' '.join(word_strings)

def zhuang_mode_ipa_ipa(sentence):
    word_strings = []
    for prefix, core, suffix in split_words(sentence):
        word_parts = []
        if prefix:
            word_parts.append(prefix)
        if is_zhuang_word(core):
            processed = process_word(core)
            converted = []
            for item in processed:
                if item in initials:
                    converted.append(initials[item])
                elif item in finals:
                    converted.append(finals[item])
                elif tone_marker in item:
                    tone_char = item.replace(tone_marker, '')
                    if tone_char in tone_symbols_to_numbers:
                        num = tone_symbols_to_numbers[tone_char]
                    elif tone_char in stop_tone_endings:
                        num = stop_tone_endings[tone_char]
                    else:
                        num = tone_char
                    converted.append(ipa_tone_symbols.get(num, ipa_tone_symbols["1"]))
                else:
                    converted.append(item)
            word_parts.extend(converted)
        else:
            word_parts.append(core)
        if suffix:
            word_parts.append(suffix)
        word_str = ' '.join(filter(None, word_parts))
        word_strings.append(word_str)
    return ' '.join(word_strings)

def zhuang_mode_ipa_num(sentence):
    word_strings = []
    for prefix, core, suffix in split_words(sentence):
        word_parts = []
        if prefix:
            word_parts.append(prefix)
        if is_zhuang_word(core):
            processed = process_word(core)
            converted = []
            for item in processed:
                if item in initials:
                    converted.append(initials[item])
                elif item in finals:
                    converted.append(finals[item])
                elif tone_marker in item:
                    tone_char = item.replace(tone_marker, '')
                    if tone_char in tone_symbols_to_numbers:
                        num = tone_symbols_to_numbers[tone_char]
                    elif tone_char in stop_tone_endings:
                        num = stop_tone_endings[tone_char]
                    else:
                        num = '1'
                    converted.append(f'{num}{tone_marker}')
                else:
                    converted.append(item)
            word_parts.extend(converted)
        else:
            word_parts.append(core)
        if suffix:
            word_parts.append(suffix)
        word_str = ' '.join(filter(None, word_parts))
        word_strings.append(word_str)
    return ' '.join(word_strings)

def zhuang_mode_py_ipa(sentence):
    word_strings = []
    for prefix, core, suffix in split_words(sentence):
        word_parts = []
        if prefix:
            word_parts.append(prefix)
        if is_zhuang_word(core):
            processed = process_word(core)
            converted = []
            for item in processed:
                if item in initials or item in finals:
                    converted.append(item)
                elif tone_marker in item:
                    tone_char = item.replace(tone_marker, '')
                    if tone_char in tone_symbols_to_numbers:
                        num = tone_symbols_to_numbers[tone_char]
                    elif tone_char in stop_tone_endings:
                        num = stop_tone_endings[tone_char]
                    else:
                        num = tone_char
                    converted.append(ipa_tone_symbols.get(num, ipa_tone_symbols["1"]))
                else:
                    converted.append(item)
            word_parts.extend(converted)
        else:
            word_parts.append(core)
        if suffix:
            word_parts.append(suffix)
        word_str = ' '.join(filter(None, word_parts))
        word_strings.append(word_str)
    return ' '.join(word_strings)

def zhuang_mode_py_num(sentence):
    word_strings = []
    for prefix, core, suffix in split_words(sentence):
        word_parts = []
        if prefix:
            word_parts.append(prefix)
        if is_zhuang_word(core):
            processed = process_word(core)
            converted = []
            for item in processed:
                if item in initials or item in finals:
                    converted.append(item)
                elif tone_marker in item:
                    tone_char = item.replace(tone_marker, '')
                    if tone_char in tone_symbols_to_numbers:
                        num = tone_symbols_to_numbers[tone_char]
                    elif tone_char in stop_tone_endings:
                        num = stop_tone_endings[tone_char]
                    else:
                        num = '1'
                    converted.append(f'{num}{tone_marker}')
                else:
                    converted.append(item)
            word_parts.extend(converted)
        else:
            word_parts.append(core)
        if suffix:
            word_parts.append(suffix)
        word_str = ' '.join(filter(None, word_parts))
        word_strings.append(word_str)
    return ' '.join(word_strings)

def zhuang_to_ipa_with_tone(sentence):
    """
    壮文字母/音节直接转国际音标，原有空格保留，不添加任何新空格。
    """
    result = []
    for prefix, core, suffix in split_words(sentence):
        ipa_word = ''
        if is_zhuang_word(core):
            word = core
            max_loops = len(word) * 2 + 10
            loop_count = 0
            while word and loop_count < max_loops:
                loop_count += 1
                result_syll = parse_syllable(word)
                initial, final, tone, leftover = result_syll
                if initial or final:
                    if initial and initial in initials:
                        ipa_word += initials[initial]
                    if final and final in finals:
                        ipa_word += finals[final]
                    if tone:
                        if tone in tone_symbols_to_numbers:
                            num = tone_symbols_to_numbers[tone]
                        elif tone in stop_tone_endings:
                            num = stop_tone_endings[tone]
                        else:
                            num = tone
                        ipa_word += ipa_tone_symbols.get(num, ipa_tone_symbols["1"])
                    word = leftover
                else:
                    ipa_word += word
                    break
        else:
            ipa_word = core
        result.append(prefix + ipa_word + suffix)
    return ' '.join(result)

# ===================== 文件处理 =====================
def process_file(input_path, output_dir, failed_words_file=None):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_files = {
        'ipa_ipa': os.path.join(output_dir, f"{base_name}_ipa_ipa.txt"),
        'ipa_num': os.path.join(output_dir, f"{base_name}_ipa_num.txt"),
        'py_ipa': os.path.join(output_dir, f"{base_name}_py_ipa.txt"),
        'py_num': os.path.join(output_dir, f"{base_name}_py_num.txt"),
        'py_py': os.path.join(output_dir, f"{base_name}_py_py.txt"),
        'py_pyhash': os.path.join(output_dir, f"{base_name}_py_pyhash.txt"),
        'ipa_with_tone': os.path.join(output_dir, f"{base_name}_ipa_with_tone.txt"),
    }
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    for zxmode, path in output_files.items():
        with open(path, 'w', encoding='utf-8') as fout:
            for line in tqdm(lines, desc=f"Processing {zxmode} for {base_name}"):
                line = line.strip()
                if not line:
                    continue
                fields = line.split('|')
                if len(fields) == 3:
                    filename, speaker, text = fields
                    cleaned_text = text.strip('"')
                    cleaned_line = f"{filename}|{speaker}|{convert_zhuang(cleaned_text, zxmode)}\n"
                elif len(fields) == 2:
                    filename, text = fields
                    cleaned_text = text.strip('"')
                    cleaned_line = f"{filename}|{convert_zhuang(cleaned_text, zxmode)}\n"
                else:
                    continue
                fout.write(cleaned_line)
    if failed_words_file:
        with open(failed_words_file, "w", encoding="utf-8") as f:
            for word in sorted(failed_words):
                f.write(word + "\n")
    return list(output_files.values())

def generate_cleaned_files(raw_txt_path, output_dir):
    base_name = os.path.splitext(os.path.basename(raw_txt_path))[0]
    modes = ['ipa_ipa', 'ipa_num', 'py_ipa', 'py_num', 'py_py', 'py_pyhash', 'ipa_with_tone']
    with open(raw_txt_path, 'r', encoding='utf-8') as f:
        preview = [line.strip() for line in f if line.strip()]
    col_count = None
    for line in preview[:10]:
        fields = line.split('|')
        if len(fields) == 3:
            col_count = 3
            break
        elif len(fields) == 2:
            col_count = 2
            break
    if col_count is None:
        raise ValueError("文件内容无法识别单双说话人格式！")

    for zxmode in modes:
        cleaned_path = os.path.join(output_dir, f"{base_name}_{zxmode}.txt.cleaned")
        with open(raw_txt_path, 'r', encoding='utf-8') as f_in:
            all_lines = [line for line in f_in if line.strip()]
        with open(raw_txt_path, 'r', encoding='utf-8') as f_in, \
             open(cleaned_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, total=len(all_lines), desc=f"Cleaning {zxmode} for {base_name}"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if col_count == 3 and len(parts) == 3:
                    filename, speaker, text = parts
                    cleaned_text = text.strip('"')
                    line_out = f"{filename}|{speaker}|{convert_zhuang(cleaned_text, zxmode)}\n"
                    f_out.write(line_out)
                elif col_count == 2 and len(parts) == 2:
                    filename, text = parts
                    cleaned_text = text.strip('"')
                    line_out = f"{filename}|{convert_zhuang(cleaned_text, zxmode)}\n"
                    f_out.write(line_out)

def batch_process(input_dir, output_dir):
    target_files = {'train.txt', 'val.txt', 'test.txt'}
    all_files = os.listdir(input_dir)
    for filename in all_files:
        if filename not in target_files:
            print(f"跳过非目标文件: {filename}")
            continue
        input_path = os.path.join(input_dir, filename)
        if not os.path.isfile(input_path):
            print(f"跳过目录: {filename}")
            continue
        failed_words_file = os.path.join(output_dir, f"{filename}_failed_words.txt")
        output_files = process_file(input_path, output_dir, failed_words_file=failed_words_file)
        for file_path in output_files:
            generate_cleaned_files(file_path, output_dir)

def test_convert_zhuang():
    print("==== 壮语多模式分词/声调映射测试 ====")
    test_cases = [
        ("dong gen caux cun gen", {
            "ipa_ipa": "t oːŋ ˧˩ k eːn ˨ ɕ aːu ˥ ɕ un ˨ k eːn ˨",
            "ipa_num": "t oːŋ 8# k eːn 1# ɕ aːu 4# ɕ un 1# k eːn 1#",
            "py_ipa": "d ong ˧˩ g en ˨ c au ˥ c un ˨ g en ˨",
            "py_num": "d ong 8# g en 1# c au 4# c un 1# g en 1#",
            "py_py": "d ong g en c au x c un g en",
            "py_pyhash": "d ong g en c au x# c un g en",
            "ipa_with_tone": "toːŋ˧˩ keːn˨ ɕaːu˥ ɕun˨ keːn˨",
        }),
    ]
    modes = ['ipa_ipa', 'ipa_num', 'py_ipa', 'py_num', 'py_py', 'py_pyhash', 'ipa_with_tone']
    for sentence, expected in test_cases:
        print(f"\n原句: {sentence}")
        for mode in modes:
            result = convert_zhuang(sentence, mode)
            print(f"{mode:<12}: {result}")
            assert result == expected[mode], f"{mode} 不匹配，期望: {expected[mode]}，实际: {result}"

# if __name__ == "__main__":
#     test_convert_zhuang()
#     # input_dir = r"E:\WorkSpace\Python\ZhuangYu\datasets\05_26\datasets"
#     # output_dir = r"E:\WorkSpace\Python\ZhuangYu\datasets\05_26\output_all_new"
#     # os.makedirs(output_dir, exist_ok=True)
#     # batch_process(input_dir, output_dir)
if __name__ == "__main__":
    test_convert_zhuang()

    input_dir = r"E:\WorkSpace\Python\ZhuangYu\datasets\05_30\multi_datasets"
    output_dir = r"E:\WorkSpace\Python\ZhuangYu\datasets\05_30\multi_datasets\output_all_new"
    os.makedirs(output_dir, exist_ok=True)
    target_files = {'train.txt', 'val.txt', 'test.txt'}
    all_files = os.listdir(input_dir)
    for filename in all_files:
        if filename not in target_files:
            print(f"跳过非目标文件: {filename}")
            continue
        raw_txt_path = os.path.join(input_dir, filename)
        generate_cleaned_files(raw_txt_path, output_dir)