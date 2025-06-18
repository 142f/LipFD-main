# 导入必要的库
import gzip  # 用于处理gzip压缩文件
import html  # 用于处理HTML实体
import os  # 用于操作系统相关功能，如路径操作
from functools import lru_cache  # 用于缓存函数结果，提高性能

import ftfy  # 用于修复Unicode文本中的常见问题
import regex as re  # 导入regex库，提供更强大的正则表达式功能


# 使用lru_cache缓存函数结果，避免重复计算
@lru_cache()
def default_bpe():
    """
    获取默认BPE（Byte Pair Encoding）词汇文件的路径。
    这个文件通常包含用于分词的字节对编码规则。
    """
    # 构建并返回bpe_simple_vocab_16e6.txt.gz文件的绝对路径
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


# 使用lru_cache缓存函数结果
@lru_cache()
def bytes_to_unicode():
    """
    创建一个从UTF-8字节到Unicode字符串的映射字典。
    这个映射对于可逆的BPE编码至关重要，因为它允许BPE在Unicode字符串上操作，
    同时避免了在词汇表中包含大量Unicode字符以避免未知字符（UNK）的问题。
    """
    # 初始化字节列表bs，包含可打印的ASCII字符、部分Latin-1补充字符和Latin-1标点符号
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    # 复制bs到cs，cs将用于存储对应的Unicode码点
    cs = bs[:]
    n = 0
    # 遍历所有256个可能的字节值
    for b in range(2**8):
        # 如果当前字节b不在bs中（即它是一个控制字符或不可打印字符）
        if b not in bs:
            bs.append(b)  # 将其添加到bs中
            cs.append(2**8 + n)  # 为其分配一个从256开始的新的Unicode码点
            n += 1  # 递增计数器
    # 将cs中的整数码点转换为对应的Unicode字符
    cs = [chr(n) for n in cs]
    # 返回一个字典，将字节（bs）映射到其对应的Unicode字符（cs）
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    从一个单词（表示为符号元组）中获取所有相邻符号对的集合。
    例如，对于单词 "hello"，如果表示为 ('h', 'e', 'l', 'l', 'o')，
    则返回的对包括 ('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')。

    Args:
        word (tuple): 表示单词的符号元组，每个符号可以是变长字符串。

    Returns:
        set: 包含所有相邻符号对的集合。
    """
    pairs = set()  # 初始化一个空集合用于存储符号对
    prev_char = word[0]  # 获取单词的第一个符号作为前一个字符
    # 遍历单词中从第二个符号开始的所有符号
    for char in word[1:]:
        pairs.add((prev_char, char))  # 将当前符号和前一个符号组成对并添加到集合中
        prev_char = char  # 更新前一个字符为当前字符
    return pairs  # 返回所有符号对的集合


def basic_clean(text):
    """
    对文本执行基本的清理操作，包括修复Unicode问题和解码HTML实体。

    Args:
        text (str): 待清理的原始文本。

    Returns:
        str: 清理后的文本。
    """
    text = ftfy.fix_text(text)  # 使用ftfy修复文本中的Unicode问题
    text = html.unescape(html.unescape(text))  # 解码HTML实体，可能需要两次解码以处理双重编码
    return text.strip()  # 移除文本两端的空白字符并返回


def whitespace_clean(text):
    """
    清理文本中的多余空白字符，将连续的空白字符替换为单个空格，并移除文本两端的空白。

    Args:
        text (str): 待清理的文本。

    Returns:
        str: 清理空白字符后的文本。
    """
    text = re.sub(r'\s+', ' ', text)  # 使用正则表达式将一个或多个空白字符替换为单个空格
    text = text.strip()  # 移除文本两端的空白字符
    return text  # 返回清理后的文本


class SimpleTokenizer(object):
    """
    SimpleTokenizer类实现了基于字节对编码（BPE）的文本分词器。
    它负责将原始文本转换为模型可以理解的token ID序列，以及将token ID序列解码回文本。
    """
    def __init__(self, bpe_path: str = default_bpe()):
        """
        初始化SimpleTokenizer。

        Args:
            bpe_path (str): BPE词汇文件的路径，默认为default_bpe()返回的路径。
        """
        # 初始化字节编码器和解码器，用于UTF-8字节和Unicode字符之间的转换
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # 读取BPE合并规则文件
        # 打开gzip压缩的BPE文件，读取内容，解码为UTF-8字符串，并按行分割
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        # 提取BPE合并规则的核心部分，跳过文件头和特殊token
        merges = merges[1:49152 - 256 - 2 + 1] # 49152是词汇表大小，256是字节，2是特殊token
        # 将每行合并规则（例如 "a b"）转换为元组 (a, b)
        merges = [tuple(merge.split()) for merge in merges]

        # 构建词汇表
        # 初始词汇表包含所有字节对应的Unicode字符
        vocab = list(bytes_to_unicode().values())
        # 为每个初始词汇添加"</w>"后缀，表示单词的结束
        vocab = vocab + [v + '</w>' for v in vocab]
        # 将BPE合并规则生成的词汇添加到词汇表中
        for merge in merges:
            vocab.append(''.join(merge))
        # 添加特殊token到词汇表
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        # 创建编码器（token到ID的映射）和解码器（ID到token的映射）
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 存储BPE合并规则的排名，用于在BPE算法中选择合并对
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        # 初始化缓存，用于存储已处理的token，避免重复计算BPE
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}

        # 编译正则表达式，用于在文本中查找token模式
        # 该模式匹配特殊token、常见缩写（如's, 't等）、字母序列、数字序列或非空白/字母/数字字符序列
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        """
        对单个token执行字节对编码（BPE）算法。
        它会根据预定义的合并规则，逐步合并token中的字符对，直到无法再合并或达到最小单元。

        Args:
            token (str): 待编码的单个token字符串。

        Returns:
            str: 经过BPE编码后的token字符串，其中子词之间用空格分隔。
        """
        # 如果token已在缓存中，直接返回缓存结果
        if token in self.cache:
            return self.cache[token]

        # 将token转换为字符元组，并在最后一个字符后添加'</w>'，表示单词结束
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        # 获取初始的字符对集合
        pairs = get_pairs(word)

        # 如果没有字符对（例如，单字符token），直接返回添加了'</w>'的token
        if not pairs:
            return token + '</w>'

        # 循环执行BPE合并，直到无法再合并
        while True:
            # 找到具有最高优先级的（即bpe_ranks中值最小的）字符对进行合并
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            # 如果找不到可合并的字符对，则退出循环
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram  # 获取要合并的两个字符
            new_word = []  # 用于构建合并后的新单词
            i = 0
            # 遍历当前单词的字符，执行合并操作
            while i < len(word):
                try:
                    # 查找第一个字符的下一个出现位置
                    j = word.index(first, i)
                    new_word.extend(word[i:j])  # 将第一个字符之前的部分添加到新单词中
                    i = j
                except:
                    new_word.extend(word[i:])  # 如果找不到，将剩余部分添加到新单词中
                    break

                # 如果当前位置是第一个字符，并且下一个字符是第二个字符，则执行合并
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)  # 添加合并后的新字符
                    i += 2  # 跳过已合并的两个字符
                else:
                    new_word.append(word[i])  # 否则，添加当前字符
                    i += 1
            new_word = tuple(new_word)  # 将新单词转换为元组形式
            word = new_word  # 更新单词为合并后的结果
            # 如果单词只剩一个字符（已完全合并），则退出循环
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)  # 否则，重新计算字符对
        word = ' '.join(word)  # 将最终的子词用空格连接起来
        self.cache[token] = word  # 将结果存入缓存
        return word  # 返回BPE编码后的token

    def encode(self, text):
        """
        将给定的文本编码为BPE token ID序列。
        这个过程包括文本清理、查找token、字节编码和BPE分词。

        Args:
            text (str): 待编码的原始文本。

        Returns:
            list: 文本对应的BPE token ID列表。
        """
        bpe_tokens = []  # 初始化一个空列表，用于存储BPE token ID
        # 对文本进行基本清理和空白字符清理，并转换为小写
        text = whitespace_clean(basic_clean(text)).lower()
        # 使用预编译的正则表达式查找文本中的所有token
        for token in re.findall(self.pat, text):
            # 将token中的每个字节编码为对应的Unicode字符（通过byte_encoder）
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 对字节编码后的token执行BPE分词，并将分词结果（子词）转换为对应的ID，然后添加到列表中
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens  # 返回BPE token ID列表

    def decode(self, tokens):
        """
        将BPE token ID序列解码回原始文本。
        这个过程是编码的逆过程，包括将ID转换为token，字节解码和移除特殊标记。

        Args:
            tokens (list): 待解码的BPE token ID列表。

        Returns:
            str: 解码后的文本字符串。
        """
        # 将token ID列表转换为对应的token字符串，并连接起来
        text = ''.join([self.decoder[token] for token in tokens])
        # 将token字符串中的Unicode字符解码回字节，然后将字节解码为UTF-8字符串，处理替换错误，并移除'</w>'标记
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text  # 返回解码后的文本
