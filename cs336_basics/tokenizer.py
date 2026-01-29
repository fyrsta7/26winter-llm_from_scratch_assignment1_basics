"""
BPE Tokenizer implementation for CS336 Assignment 1
"""
import regex as re  # Use regex library for Unicode property support
from collections import defaultdict, Counter
from typing import Iterator, Iterable
import json
import os


# GPT-2 pre-tokenization pattern
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str, special_tokens: list[str] | None = None) -> list[str]:
    """
    Pre-tokenize text using GPT-2 pattern.
    Split by special tokens first, then apply pre-tokenization.
    
    Args:
        text: Input text to pre-tokenize
        special_tokens: List of special tokens that should not be split
        
    Returns:
        List of pre-tokenized strings
    """
    if not text:
        return []
    
    # If we have special tokens, split by them first
    if special_tokens:
        # Create regex pattern that matches any special token
        # Sort by length (descending) to match longer tokens first
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        # Escape special regex characters
        escaped_tokens = [re.escape(token) for token in sorted_tokens]
        special_pattern = '|'.join(escaped_tokens)
        
        # Split by special tokens while keeping them
        parts = re.split(f'({special_pattern})', text)
        
        result = []
        for part in parts:
            if not part:
                continue
            # If this part is a special token, keep it as-is
            if part in special_tokens:
                result.append(part)
            else:
                # Apply GPT-2 pre-tokenization
                matches = GPT2_PATTERN.findall(part)
                result.extend(matches)
        return result
    else:
        # No special tokens, just apply GPT-2 pattern
        return GPT2_PATTERN.findall(text)


def get_pair_counts(pretokens: list[list[bytes]]) -> dict[tuple[bytes, bytes], int]:
    """
    Count all adjacent pairs in pre-tokenized sequences.
    
    Args:
        pretokens: List of pre-token sequences (each is a list of bytes)
        
    Returns:
        Dictionary mapping pairs to their counts
    """
    pair_counts = defaultdict(int)
    
    for pretoken_seq in pretokens:
        if len(pretoken_seq) < 2:
            continue
        for i in range(len(pretoken_seq) - 1):
            pair = (pretoken_seq[i], pretoken_seq[i + 1])
            pair_counts[pair] += 1
    
    return pair_counts


def merge_pair_fast(pretokens: list[list[bytes]], pair: tuple[bytes, bytes], 
                     pair_counts: dict[tuple[bytes, bytes], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Merge all occurrences of a pair in pre-tokenized sequences and update pair counts incrementally.
    
    Args:
        pretokens: List of pre-token sequences (modified in-place)
        pair: Pair to merge
        pair_counts: Current pair counts (modified in-place)
        
    Returns:
        Updated pair counts
    """
    merged_token = pair[0] + pair[1]
    
    for i, pretoken_seq in enumerate(pretokens):
        if len(pretoken_seq) < 2:
            continue
        
        new_seq = []
        j = 0
        while j < len(pretoken_seq):
            # Check if current and next form the pair
            if j < len(pretoken_seq) - 1 and (pretoken_seq[j], pretoken_seq[j + 1]) == pair:
                # Update pair counts for affected pairs
                # Remove old pairs
                if j > 0:
                    old_pair = (pretoken_seq[j - 1], pretoken_seq[j])
                    pair_counts[old_pair] -= 1
                    if pair_counts[old_pair] == 0:
                        del pair_counts[old_pair]
                
                if j + 2 < len(pretoken_seq):
                    old_pair = (pretoken_seq[j + 1], pretoken_seq[j + 2])
                    pair_counts[old_pair] -= 1
                    if pair_counts[old_pair] == 0:
                        del pair_counts[old_pair]
                
                # Add new pairs
                if len(new_seq) > 0:
                    new_pair = (new_seq[-1], merged_token)
                    pair_counts[new_pair] += 1
                
                if j + 2 < len(pretoken_seq):
                    new_pair = (merged_token, pretoken_seq[j + 2])
                    pair_counts[new_pair] += 1
                
                # Merge the pair
                new_seq.append(merged_token)
                j += 2
            else:
                new_seq.append(pretoken_seq[j])
                j += 1
        
        pretokens[i] = new_seq
    
    # Remove the merged pair from counts
    if pair in pair_counts:
        del pair_counts[pair]
    
    return pair_counts


def train_bpe(
    corpus_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer.
    
    Args:
        corpus_path: Path to training corpus
        vocab_size: Desired vocabulary size (including special tokens)
        special_tokens: List of special tokens to add
        
    Returns:
        vocab: Mapping from token ID to token bytes
        merges: List of merge operations (in order)
    """
    # Initialize vocabulary with 256 byte tokens
    vocab = {i: bytes([i]) for i in range(256)}
    
    # Add special tokens
    for special_token in special_tokens:
        special_bytes = special_token.encode('utf-8')
        vocab[len(vocab)] = special_bytes
    
    # Read corpus and pre-tokenize
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Pre-tokenize the text
    pretokens_str = pretokenize(text, special_tokens)
    
    # Convert pre-tokens to byte sequences and count their frequencies
    # This way we can process unique patterns instead of all instances
    pretoken_counts = Counter()
    for pretoken_str in pretokens_str:
        if pretoken_str in special_tokens:
            # Skip special tokens in BPE merging
            continue
        else:
            # Convert to bytes and split into individual byte tokens
            byte_seq = pretoken_str.encode('utf-8')
            token_tuple = tuple(bytes([b]) for b in byte_seq)
            pretoken_counts[token_tuple] += 1
    
    merges = []
    
    # Initial pair counts
    pair_counts = Counter()
    for pretoken, count in pretoken_counts.items():
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_counts[pair] += count
    
    # Perform merges until we reach desired vocab size
    while len(vocab) < vocab_size:
        if not pair_counts:
            # No more pairs to merge
            break
        
        # Find most frequent pair (ties broken by lexicographic order)
        # Note: We want the lexicographically LARGEST pair in case of tie
        max_count = max(pair_counts.values())
        most_frequent_pairs = [pair for pair, count in pair_counts.items() if count == max_count]
        best_pair = max(most_frequent_pairs)  # Lexicographically largest
        
        # Merge the best pair in all pretokens and update pair counts incrementally
        new_pretoken_counts = Counter()
        merged_token = best_pair[0] + best_pair[1]
        
        for pretoken, count in pretoken_counts.items():
            # Check if this pretoken contains the pair
            if best_pair[0] not in pretoken or best_pair[1] not in pretoken:
                # No change needed
                new_pretoken_counts[pretoken] += count
                continue
            
            # Remove old pair counts for this pretoken
            for i in range(len(pretoken) - 1):
                old_pair = (pretoken[i], pretoken[i + 1])
                pair_counts[old_pair] -= count
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
            
            # Merge the pair in this pretoken
            new_pretoken = []
            i = 0
            while i < len(pretoken):
                if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == best_pair:
                    new_pretoken.append(merged_token)
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1
            
            new_pretoken_tuple = tuple(new_pretoken)
            new_pretoken_counts[new_pretoken_tuple] += count
            
            # Add new pair counts for this pretoken
            for i in range(len(new_pretoken) - 1):
                new_pair = (new_pretoken[i], new_pretoken[i + 1])
                pair_counts[new_pair] += count
        
        pretoken_counts = new_pretoken_counts
        
        # Add merged token to vocabulary
        vocab[len(vocab)] = merged_token
        
        # Record the merge
        merges.append(best_pair)
    
    return vocab, merges


class Tokenizer:
    """BPE Tokenizer"""
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab: Mapping from token ID to token bytes
            merges: List of BPE merges in order
            special_tokens: List of special tokens
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Create reverse vocab for encoding
        self.token_to_id = {token: idx for idx, token in vocab.items()}
        
        # Create merge priority dictionary
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None
    ):
        """
        Load tokenizer from vocab and merges files.
        
        Args:
            vocab_filepath: Path to vocabulary JSON file
            merges_filepath: Path to merges text file
            special_tokens: List of special tokens
            
        Returns:
            Tokenizer instance
        """
        # Load vocab
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        
        # Convert vocab (assuming it's in GPT-2 format or similar)
        # This depends on the file format
        vocab = {}
        for token_str, token_id in vocab_dict.items():
            vocab[token_id] = token_str.encode('utf-8')
        
        # Load merges
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ' ' in line:
                    token1, token2 = line.split(' ', 1)
                    merges.append((token1.encode('utf-8'), token2.encode('utf-8')))
        
        return cls(vocab, merges, special_tokens)
    
    def _apply_merges(self, byte_tokens: list[bytes]) -> list[bytes]:
        """
        Apply BPE merges to a sequence of byte tokens.
        
        Args:
            byte_tokens: List of byte tokens
            
        Returns:
            List of merged byte tokens
        """
        if len(byte_tokens) < 2:
            return byte_tokens
        
        while True:
            # Find the pair with highest priority (lowest index in merges list)
            best_pair = None
            best_priority = float('inf')
            best_pos = -1
            
            for i in range(len(byte_tokens) - 1):
                pair = (byte_tokens[i], byte_tokens[i + 1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair
                        best_pos = i
            
            if best_pair is None:
                # No more merges possible
                break
            
            # Merge the pair at best_pos
            new_tokens = []
            i = 0
            while i < len(byte_tokens):
                if i == best_pos:
                    new_tokens.append(byte_tokens[i] + byte_tokens[i + 1])
                    i += 2
                elif i == best_pos + 1:
                    # Already merged
                    i += 1
                else:
                    new_tokens.append(byte_tokens[i])
                    i += 1
            
            byte_tokens = new_tokens
            
            if len(byte_tokens) < 2:
                break
        
        return byte_tokens
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        # Pre-tokenize
        pretokens = pretokenize(text, self.special_tokens)
        
        token_ids = []
        for pretoken in pretokens:
            # Check if it's a special token
            if pretoken in self.special_tokens:
                special_bytes = pretoken.encode('utf-8')
                if special_bytes in self.token_to_id:
                    token_ids.append(self.token_to_id[special_bytes])
                continue
            
            # Convert to bytes and split into byte tokens
            byte_seq = pretoken.encode('utf-8')
            byte_tokens = [bytes([b]) for b in byte_seq]
            
            # Apply BPE merges
            merged_tokens = self._apply_merges(byte_tokens)
            
            # Convert to IDs
            for token in merged_tokens:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    # This shouldn't happen with a properly trained tokenizer
                    # Fall back to byte-level encoding
                    for b in token:
                        token_ids.append(b)
        
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings, yielding token IDs.
        Memory-efficient for large files.
        
        Args:
            iterable: Iterable of strings (e.g., file object)
            
        Yields:
            Token IDs
        """
        for chunk in iterable:
            token_ids = self.encode(chunk)
            yield from token_ids
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        # Convert IDs to bytes
        byte_sequence = b''
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequence += self.vocab[token_id]
        
        # Decode bytes to string, replacing invalid sequences with U+FFFD
        return byte_sequence.decode('utf-8', errors='replace')
