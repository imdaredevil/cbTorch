from .tokenizer import Tokenizer
from typing import List, Tuple
from pyllist import dllist
import tqdm

class BPETokenizer(Tokenizer):
    def __init__(self, unicode_encoding = "utf-8"):
        super().__init__()
        self.unicode_encoding = unicode_encoding
        self.initialize_vocab()
        
    def initialize_vocab(self):
        super().initialize_vocab()
        for num in range(256):
            self.vocabulary[num.to_bytes(1, "little")] = num
            self.tokens.append(num.to_bytes(1, "little"))
        self.max_byte = 1
        self.vocab_size = 256

    def _ispair_valid(self, pair: Tuple[bytes]):
        first = pair[0].decode(self.unicode_encoding, "ignore").strip()
        second = pair[1].decode(self.unicode_encoding, "ignore").strip()
        word = first + second
        if word.isalpha() or word.isnumeric():
            return True
        for char in word:
            if char.isalpha() or char.isnumeric():
                return False
        return True

    # def _form_bytepair_count_map(self, data_list: List[dllist]):
    #     bytepair_count_map = {}
    #     max_count_pair = (b"", 0)
    #     for sentence in data_list:
    #         num_bytes = sentence.size
    #         curr_node = sentence.first
    #         for _ in range(num_bytes - 1):
    #             next_node = curr_node.next
    #             pair = curr_node.value + next_node.value
    #             if not self._ispair_valid(pair):
    #                 continue
    #             if pair not in bytepair_count_map:
    #                 bytepair_count_map[pair] = 0
    #             bytepair_count_map[pair] += 1
    #             curr_node = next_node
    #     for pair, count in bytepair_count_map.items():
    #         if count > max_count_pair[1]:
    #             max_count_pair = (pair, count)
    #     return bytepair_count_map, max_count_pair

    def _update_bytepair_count_map(self, data_list: List[dllist], new_word: bytes = None):
        count_map = {}
        max_count_pair = (b"", 0)
        for sentence in data_list:
            curr_node = sentence.first
            while True:
                next_node = curr_node.next
                if next_node is None:
                    break
                pair = curr_node.value + next_node.value
                if pair == new_word:
                    new_node = sentence.insert(new_word, after = curr_node)
                    sentence.remove(next_node)
                    sentence.remove(curr_node)
                    curr_node = new_node
                    continue
                if self._ispair_valid((curr_node.value, next_node.value)):
                    if pair not in count_map:
                        count_map[pair] = 0
                    count_map[pair] += 1
                curr_node = next_node
        for pair, count in count_map.items():
            if count > max_count_pair[1]:
                max_count_pair = (pair, count)
        return max_count_pair
                    
    def tokenize(self, data: List[str], to_list = True):
        linkedlist_data = []
        for sentence in tqdm.tqdm(data):
            sentence_bytes = sentence.encode(self.unicode_encoding)
            sentence_bytes = [bytes([byte]) for byte in sentence_bytes]
            curr_ll = dllist(sentence_bytes)
            num_byte = self.max_byte
            while num_byte > 0:
                # print(curr_ll, num_byte)
                curr_node = curr_ll.first
                while True:
                    if curr_node is None:
                        break
                    if len(curr_node.value) > 1:
                        curr_node = curr_node.next
                        continue
                    start = curr_node
                    end = start
                    sequence = b""
                    for _ in range(num_byte):
                        if end is None:
                            break
                        sequence += end.value
                        end = end.next
                    if len(sequence) < num_byte:
                        break
                    if sequence in self.vocabulary:
                        prev = start.prev
                        for _ in range(num_byte):
                            new_start = start.next
                            curr_ll.remove(start)
                            start = new_start
                        if prev is None:
                            curr_node = curr_ll.appendleft(sequence)
                        else:
                            curr_node = curr_ll.insert(sequence, after = prev)
                    curr_node = curr_node.next
                    # print(sequence, curr_ll, start)
                num_byte -= 1
            linkedlist_data.append(curr_ll)
        if to_list:
            linkedlist_data = [[x.value for x in sentence.iternodes()] for sentence in linkedlist_data]
        return linkedlist_data

    def train(self, data: List[str], max_vocab_size = 50, reinitialize= True):
        if reinitialize:
            self.initialize_vocab()
        if (self.vocab_size >= max_vocab_size):
            return
        linkedlist_data = self.tokenize(data, to_list = False)
        new_word = None
        for _ in tqdm.tqdm(range(self.vocab_size, max_vocab_size)):
            max_count_pair = self._update_bytepair_count_map(linkedlist_data, new_word)
            if max_count_pair[1] == 0:
                print("WARN: reached end of vocab")
                return
            new_word = max_count_pair[0]
            self.max_byte = max(len(new_word), self.max_byte)
            self.vocabulary[new_word] = self.vocab_size
            self.vocab_size += 1
            self.tokens.append(new_word)
    
    def encode(self, data: List[str]):
        tokens = self.tokenize(data)
        tokens = [
            [self.vocabulary[token] for token in sentence]
            for sentence in tokens
        ]
        return tokens
    
    def decode(self, tokens, to_str = True):
        sentence_bytes = [
            [self.tokens[token] for token in sentence]
            for sentence in tokens
        ]
        if to_str:
            sentence_bytes = [
                b"".join(sentence) for sentence in sentence_bytes
            ]
            sentence_bytes = [
                sentence.decode(self.unicode_encoding) for sentence in sentence_bytes
            ]
        return sentence_bytes   
            
        
                