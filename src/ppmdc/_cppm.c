#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define ALPHABET_SIZE 257
#define EOF_SYMBOL 256
#define MAX_CONTEXT_TOTAL (1U << 15)

#define STATE_BITS 32
#define FULL_RANGE (1ULL << STATE_BITS)
#define HALF_RANGE (FULL_RANGE >> 1)
#define QUARTER_RANGE (HALF_RANGE >> 1)
#define THREE_QUARTER_RANGE (QUARTER_RANGE * 3)
#define MASK (FULL_RANGE - 1)

#define MASK_WORDS 5

#define HEADER_TAG 0
#define SEQUENCE_TAG 1
#define SEQUENCE_REPEAT_TAG 2
#define SEQUENCE_GROUP_TAG 3
#define SEQUENCE_PACKED_TAG 4
#define SEQUENCE_GROUP_PACKED_TAG 5

#define LITERAL_RUN_TAG 0
#define MATCH_RUN_TAG 1
#define REVERSE_COMPLEMENT_RUN_TAG 2

#define NEWLINE_NONE 0
#define NEWLINE_LF 1
#define NEWLINE_CRLF 2
#define NEWLINE_CR 3

#define REPEAT_KMER_SIZE 8
#define REPEAT_MIN_MATCH 12
#define REPEAT_MAX_CANDIDATES 32

typedef struct ContextNode ContextNode;

typedef struct {
    uint16_t symbol;
    uint32_t count;
} CountEntry;

typedef struct {
    uint16_t symbol;
    ContextNode *node;
} ChildEntry;

struct ContextNode {
    CountEntry *counts;
    ChildEntry *children;
    uint16_t count_len;
    Py_ssize_t count_cap;
    uint16_t child_len;
    Py_ssize_t child_cap;
    uint32_t total;
    uint64_t symbol_mask[MASK_WORDS];
};

typedef struct {
    ContextNode **items;
    Py_ssize_t len;
    Py_ssize_t cap;
} NodeArena;

typedef struct {
    uint8_t *data;
    Py_ssize_t len;
    Py_ssize_t cap;
    uint8_t current_byte;
    int bits_filled;
} BitWriter;

typedef struct {
    const uint8_t *data;
    Py_ssize_t len;
    Py_ssize_t pos;
    uint8_t current_byte;
    int bits_remaining;
} BitReader;

typedef struct {
    BitWriter *output;
    uint32_t low;
    uint32_t high;
    uint32_t pending_bits;
} ArithmeticEncoder;

typedef struct {
    BitReader *input;
    uint32_t low;
    uint32_t high;
    uint32_t code;
} ArithmeticDecoder;

typedef struct {
    int order;
    NodeArena arena;
    ContextNode *root;
    ContextNode **contexts;
    ContextNode **next_contexts;
    int context_len;
} PPMModel;

typedef struct {
    uint8_t *data;
    Py_ssize_t len;
    Py_ssize_t cap;
} ByteBuffer;

typedef struct {
    uint64_t *data;
    Py_ssize_t len;
    Py_ssize_t cap;
} UInt64Buffer;

typedef struct {
    int is_header;
    const uint8_t *body;
    Py_ssize_t body_len;
    uint8_t newline;
} FastaLineView;

typedef struct {
    FastaLineView *items;
    Py_ssize_t len;
    Py_ssize_t cap;
} FastaLineArray;

typedef struct {
    uint64_t key;
    Py_ssize_t positions[32];
    uint8_t used;
    uint8_t len;
    uint8_t start;
} KmerEntry;

typedef struct {
    KmerEntry *entries;
    Py_ssize_t size;
    Py_ssize_t used;
} KmerIndex;


static int
ensure_capacity(void **buffer, Py_ssize_t *cap, Py_ssize_t item_size, Py_ssize_t needed)
{
    void *grown = NULL;
    Py_ssize_t next_cap = 0;

    if (*cap >= needed) {
        return 0;
    }

    next_cap = (*cap == 0) ? 8 : *cap;
    while (next_cap < needed) {
        next_cap *= 2;
    }

    grown = PyMem_Realloc(*buffer, item_size * next_cap);
    if (grown == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    *buffer = grown;
    *cap = next_cap;
    return 0;
}


static void
byte_buffer_init(ByteBuffer *buffer)
{
    buffer->data = NULL;
    buffer->len = 0;
    buffer->cap = 0;
}


static void
byte_buffer_free(ByteBuffer *buffer)
{
    PyMem_Free(buffer->data);
    buffer->data = NULL;
    buffer->len = 0;
    buffer->cap = 0;
}


static int
byte_buffer_reserve(ByteBuffer *buffer, Py_ssize_t needed)
{
    return ensure_capacity((void **)&buffer->data, &buffer->cap, sizeof(uint8_t), needed);
}


static int
byte_buffer_append_byte(ByteBuffer *buffer, uint8_t value)
{
    if (byte_buffer_reserve(buffer, buffer->len + 1) < 0) {
        return -1;
    }
    buffer->data[buffer->len++] = value;
    return 0;
}


static int
byte_buffer_append_bytes(ByteBuffer *buffer, const uint8_t *data, Py_ssize_t len)
{
    if (len == 0) {
        return 0;
    }
    if (byte_buffer_reserve(buffer, buffer->len + len) < 0) {
        return -1;
    }
    memcpy(buffer->data + buffer->len, data, (size_t)len);
    buffer->len += len;
    return 0;
}


static PyObject *
byte_buffer_to_bytes(const ByteBuffer *buffer)
{
    return PyBytes_FromStringAndSize((const char *)buffer->data, buffer->len);
}


static void
u64_buffer_init(UInt64Buffer *buffer)
{
    buffer->data = NULL;
    buffer->len = 0;
    buffer->cap = 0;
}


static void
u64_buffer_free(UInt64Buffer *buffer)
{
    PyMem_Free(buffer->data);
    buffer->data = NULL;
    buffer->len = 0;
    buffer->cap = 0;
}


static int
u64_buffer_append(UInt64Buffer *buffer, uint64_t value)
{
    if (ensure_capacity((void **)&buffer->data, &buffer->cap, sizeof(uint64_t), buffer->len + 1) < 0) {
        return -1;
    }
    buffer->data[buffer->len++] = value;
    return 0;
}


static void
fasta_line_array_free(FastaLineArray *array)
{
    PyMem_Free(array->items);
    array->items = NULL;
    array->len = 0;
    array->cap = 0;
}


static int
fasta_line_array_append(FastaLineArray *array, FastaLineView line)
{
    if (ensure_capacity((void **)&array->items, &array->cap, sizeof(FastaLineView), array->len + 1) < 0) {
        return -1;
    }
    array->items[array->len++] = line;
    return 0;
}


static uint8_t
complement_byte(uint8_t value)
{
    switch (value) {
        case 'A': return 'T';
        case 'B': return 'V';
        case 'C': return 'G';
        case 'D': return 'H';
        case 'G': return 'C';
        case 'H': return 'D';
        case 'K': return 'M';
        case 'M': return 'K';
        case 'N': return 'N';
        case 'R': return 'Y';
        case 'S': return 'S';
        case 'T': return 'A';
        case 'U': return 'A';
        case 'V': return 'B';
        case 'W': return 'W';
        case 'X': return 'X';
        case 'Y': return 'R';
        case '-': return '-';
        case '.': return '.';
        case '*': return '*';
        case 'a': return 't';
        case 'b': return 'v';
        case 'c': return 'g';
        case 'd': return 'h';
        case 'g': return 'c';
        case 'h': return 'd';
        case 'k': return 'm';
        case 'm': return 'k';
        case 'n': return 'n';
        case 'r': return 'y';
        case 's': return 's';
        case 't': return 'a';
        case 'u': return 'a';
        case 'v': return 'b';
        case 'w': return 'w';
        case 'x': return 'x';
        case 'y': return 'r';
        default: return value;
    }
}


static int
dna_base_code(uint8_t value)
{
    switch (value) {
        case 'A':
        case 'a':
            return 0;
        case 'C':
        case 'c':
            return 1;
        case 'G':
        case 'g':
            return 2;
        case 'T':
        case 't':
            return 3;
        default:
            return -1;
    }
}


static uint64_t
splitmix64(uint64_t value)
{
    value += UINT64_C(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}


static int
append_varint(ByteBuffer *buffer, Py_ssize_t value)
{
    uint8_t encoded[10];
    int count = 0;

    if (value < 0) {
        PyErr_SetString(PyExc_ValueError, "varint cannot encode negative values");
        return -1;
    }

    do {
        uint8_t chunk = (uint8_t)(value & 0x7F);
        value >>= 7;
        if (value != 0) {
            chunk |= 0x80;
        }
        encoded[count++] = chunk;
    } while (value != 0);

    return byte_buffer_append_bytes(buffer, encoded, count);
}


static int
read_varint(const uint8_t *data, Py_ssize_t len, Py_ssize_t *offset, Py_ssize_t *value_out)
{
    Py_ssize_t value = 0;
    int shift = 0;

    for (;;) {
        uint8_t current = 0;
        if (*offset >= len) {
            PyErr_SetString(PyExc_ValueError, "truncated varint");
            return -1;
        }
        current = data[(*offset)++];
        value |= ((Py_ssize_t)(current & 0x7F)) << shift;
        if ((current & 0x80) == 0) {
            *value_out = value;
            return 0;
        }
        shift += 7;
        if (shift > 63) {
            PyErr_SetString(PyExc_ValueError, "varint is too large");
            return -1;
        }
    }
}


static Py_ssize_t
packed_symbol_size(Py_ssize_t symbol_count)
{
    return (symbol_count + 3) / 4;
}


static int
append_packed_symbols(ByteBuffer *buffer, const uint8_t *symbols, Py_ssize_t len)
{
    Py_ssize_t offset = 0;
    while (offset < len) {
        uint8_t packed = 0;
        int count = 0;
        while (count < 4 && offset < len) {
            packed = (uint8_t)((packed << 2) | (symbols[offset] & 0x03));
            offset += 1;
            count += 1;
        }
        while (count < 4) {
            packed <<= 2;
            count += 1;
        }
        if (byte_buffer_append_byte(buffer, packed) < 0) {
            return -1;
        }
    }
    return 0;
}


static int
unpack_symbols(ByteBuffer *buffer, const uint8_t *packed, Py_ssize_t symbol_count)
{
    Py_ssize_t produced = 0;
    Py_ssize_t byte_index = 0;
    while (produced < symbol_count) {
        uint8_t current = packed[byte_index++];
        for (int shift = 6; shift >= 0 && produced < symbol_count; shift -= 2) {
            if (byte_buffer_append_byte(buffer, (uint8_t)((current >> shift) & 0x03)) < 0) {
                return -1;
            }
            produced += 1;
        }
    }
    return 0;
}


static uint64_t
hash_u64(uint64_t value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}


static int
kmer_index_init(KmerIndex *index, Py_ssize_t size)
{
    Py_ssize_t actual_size = 8;
    while (actual_size < size) {
        actual_size *= 2;
    }

    index->entries = PyMem_Calloc((size_t)actual_size, sizeof(KmerEntry));
    if (index->entries == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    index->size = actual_size;
    index->used = 0;
    return 0;
}


static void
kmer_index_free(KmerIndex *index)
{
    PyMem_Free(index->entries);
    index->entries = NULL;
    index->size = 0;
    index->used = 0;
}


static KmerEntry *
kmer_index_lookup_entry(const KmerIndex *index, uint64_t key)
{
    if (index->entries == NULL || index->size == 0) {
        return NULL;
    }
    Py_ssize_t mask = index->size - 1;
    Py_ssize_t slot = (Py_ssize_t)(hash_u64(key) & (uint64_t)mask);

    while (index->entries[slot].used) {
        if (index->entries[slot].key == key) {
            return &((KmerIndex *)index)->entries[slot];
        }
        slot = (slot + 1) & mask;
    }

    return NULL;
}


static int
kmer_index_rehash(KmerIndex *index, Py_ssize_t new_size)
{
    KmerEntry *old_entries = index->entries;
    Py_ssize_t old_size = index->size;
    Py_ssize_t old_used = index->used;

    if (kmer_index_init(index, new_size) < 0) {
        index->entries = old_entries;
        index->size = old_size;
        index->used = old_used;
        return -1;
    }

    for (Py_ssize_t old_slot = 0; old_slot < old_size; old_slot++) {
        KmerEntry *source = &old_entries[old_slot];
        if (!source->used) {
            continue;
        }
        Py_ssize_t mask = index->size - 1;
        Py_ssize_t slot = (Py_ssize_t)(hash_u64(source->key) & (uint64_t)mask);
        while (index->entries[slot].used) {
            slot = (slot + 1) & mask;
        }
        index->entries[slot] = *source;
        index->used += 1;
    }

    PyMem_Free(old_entries);
    return 0;
}


static int
kmer_index_add(KmerIndex *index, uint64_t key, Py_ssize_t position)
{
    Py_ssize_t mask = 0;
    Py_ssize_t slot = 0;
    KmerEntry *entry = NULL;

    if (index->entries == NULL) {
        if (kmer_index_init(index, 1024) < 0) {
            return -1;
        }
    }

    if ((index->used + 1) * 10 >= index->size * 7) {
        if (kmer_index_rehash(index, index->size * 2) < 0) {
            return -1;
        }
    }

    mask = index->size - 1;
    slot = (Py_ssize_t)(hash_u64(key) & (uint64_t)mask);
    while (index->entries[slot].used && index->entries[slot].key != key) {
        slot = (slot + 1) & mask;
    }

    entry = &index->entries[slot];
    if (!entry->used) {
        memset(entry, 0, sizeof(*entry));
        entry->used = 1;
        entry->key = key;
        index->used += 1;
    }

    if (entry->len < REPEAT_MAX_CANDIDATES) {
        entry->positions[entry->len++] = position;
    } else {
        entry->positions[entry->start] = position;
        entry->start = (uint8_t)((entry->start + 1U) % REPEAT_MAX_CANDIDATES);
    }

    return 0;
}


static void
mask_add(uint64_t mask[MASK_WORDS], int symbol)
{
    mask[symbol >> 6] |= (uint64_t)1 << (symbol & 63);
}


static int
mask_contains(const uint64_t mask[MASK_WORDS], int symbol)
{
    return (mask[symbol >> 6] & ((uint64_t)1 << (symbol & 63))) != 0;
}


static void
mask_or(uint64_t target[MASK_WORDS], const uint64_t source[MASK_WORDS])
{
    for (int index = 0; index < MASK_WORDS; index++) {
        target[index] |= source[index];
    }
}


static int
mask_bit_count(const uint64_t mask[MASK_WORDS])
{
    int total = 0;

    for (int index = 0; index < MASK_WORDS; index++) {
        total += __builtin_popcountll(mask[index]);
    }

    return total;
}


static int
mask_count_before_symbol(const uint64_t mask[MASK_WORDS], int symbol)
{
    int word = symbol >> 6;
    int total = 0;

    for (int index = 0; index < word; index++) {
        total += __builtin_popcountll(mask[index]);
    }

    if ((symbol & 63) != 0) {
        uint64_t lower_mask = ((uint64_t)1 << (symbol & 63)) - 1;
        total += __builtin_popcountll(mask[word] & lower_mask);
    }

    return total;
}


static void
node_arena_free(NodeArena *arena)
{
    for (Py_ssize_t index = 0; index < arena->len; index++) {
        ContextNode *node = arena->items[index];
        PyMem_Free(node->counts);
        PyMem_Free(node->children);
        PyMem_Free(node);
    }

    PyMem_Free(arena->items);
    arena->items = NULL;
    arena->len = 0;
    arena->cap = 0;
}


static ContextNode *
node_arena_new_node(NodeArena *arena)
{
    ContextNode *node = NULL;

    if (ensure_capacity((void **)&arena->items, &arena->cap, sizeof(ContextNode *), arena->len + 1) < 0) {
        return NULL;
    }

    node = PyMem_Calloc(1, sizeof(ContextNode));
    if (node == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    arena->items[arena->len++] = node;
    return node;
}


static int
context_node_rescale(ContextNode *node)
{
    uint32_t total = 0;

    for (uint16_t index = 0; index < node->count_len; index++) {
        node->counts[index].count = (node->counts[index].count + 1U) / 2U;
        total += node->counts[index].count;
    }

    node->total = total;
    return 0;
}


static int
context_node_increment(ContextNode *node, int symbol)
{
    uint16_t left = 0;
    uint16_t right = node->count_len;

    while (left < right) {
        uint16_t mid = (uint16_t)((left + right) / 2);
        if (node->counts[mid].symbol < symbol) {
            left = (uint16_t)(mid + 1);
        } else {
            right = mid;
        }
    }

    if (left < node->count_len && node->counts[left].symbol == symbol) {
        node->counts[left].count += 1;
    } else {
        if (ensure_capacity((void **)&node->counts, &node->count_cap, sizeof(CountEntry), node->count_len + 1) < 0) {
            return -1;
        }

        if (left < node->count_len) {
            memmove(
                &node->counts[left + 1],
                &node->counts[left],
                (size_t)(node->count_len - left) * sizeof(CountEntry)
            );
        }

        node->counts[left].symbol = (uint16_t)symbol;
        node->counts[left].count = 1;
        node->count_len += 1;
        mask_add(node->symbol_mask, symbol);
    }

    node->total += 1;
    if (node->total > MAX_CONTEXT_TOTAL) {
        return context_node_rescale(node);
    }

    return 0;
}


static ContextNode *
context_node_get_or_add_child(ContextNode *node, int symbol, NodeArena *arena)
{
    for (uint16_t index = 0; index < node->child_len; index++) {
        if (node->children[index].symbol == symbol) {
            return node->children[index].node;
        }
    }

    if (ensure_capacity((void **)&node->children, &node->child_cap, sizeof(ChildEntry), node->child_len + 1) < 0) {
        return NULL;
    }

    ContextNode *child = node_arena_new_node(arena);
    if (child == NULL) {
        return NULL;
    }

    node->children[node->child_len].symbol = (uint16_t)symbol;
    node->children[node->child_len].node = child;
    node->child_len += 1;
    return child;
}


static int
bit_writer_init(BitWriter *writer, Py_ssize_t initial_cap)
{
    writer->data = NULL;
    writer->len = 0;
    writer->cap = 0;
    writer->current_byte = 0;
    writer->bits_filled = 0;

    if (initial_cap > 0) {
        if (ensure_capacity((void **)&writer->data, &writer->cap, sizeof(uint8_t), initial_cap) < 0) {
            return -1;
        }
    }

    return 0;
}


static void
bit_writer_free(BitWriter *writer)
{
    PyMem_Free(writer->data);
    writer->data = NULL;
    writer->len = 0;
    writer->cap = 0;
}


static int
bit_writer_push_byte(BitWriter *writer, uint8_t value)
{
    if (ensure_capacity((void **)&writer->data, &writer->cap, sizeof(uint8_t), writer->len + 1) < 0) {
        return -1;
    }

    writer->data[writer->len++] = value;
    return 0;
}


static int
bit_writer_write(BitWriter *writer, int bit)
{
    writer->current_byte = (uint8_t)((writer->current_byte << 1) | (bit & 1));
    writer->bits_filled += 1;

    if (writer->bits_filled == 8) {
        if (bit_writer_push_byte(writer, writer->current_byte) < 0) {
            return -1;
        }
        writer->current_byte = 0;
        writer->bits_filled = 0;
    }

    return 0;
}


static int
bit_writer_finish(BitWriter *writer)
{
    if (writer->bits_filled > 0) {
        writer->current_byte = (uint8_t)(writer->current_byte << (8 - writer->bits_filled));
        if (bit_writer_push_byte(writer, writer->current_byte) < 0) {
            return -1;
        }
        writer->current_byte = 0;
        writer->bits_filled = 0;
    }

    return 0;
}


static void
bit_reader_init(BitReader *reader, const uint8_t *data, Py_ssize_t len)
{
    reader->data = data;
    reader->len = len;
    reader->pos = 0;
    reader->current_byte = 0;
    reader->bits_remaining = 0;
}


static int
bit_reader_read(BitReader *reader)
{
    if (reader->bits_remaining == 0) {
        if (reader->pos >= reader->len) {
            return 0;
        }
        reader->current_byte = reader->data[reader->pos++];
        reader->bits_remaining = 8;
    }

    reader->bits_remaining -= 1;
    return (reader->current_byte >> reader->bits_remaining) & 1;
}


static void
arithmetic_encoder_init(ArithmeticEncoder *encoder, BitWriter *writer)
{
    encoder->output = writer;
    encoder->low = 0;
    encoder->high = (uint32_t)MASK;
    encoder->pending_bits = 0;
}


static int
arithmetic_encoder_shift(ArithmeticEncoder *encoder, int bit)
{
    if (bit_writer_write(encoder->output, bit) < 0) {
        return -1;
    }

    bit ^= 1;
    while (encoder->pending_bits > 0) {
        if (bit_writer_write(encoder->output, bit) < 0) {
            return -1;
        }
        encoder->pending_bits -= 1;
    }

    return 0;
}


static int
arithmetic_encoder_update(ArithmeticEncoder *encoder, uint32_t cumulative_low, uint32_t cumulative_high, uint32_t total)
{
    uint64_t current_range = (uint64_t)encoder->high - encoder->low + 1U;

    encoder->high = (uint32_t)(encoder->low + (current_range * cumulative_high) / total - 1U);
    encoder->low = (uint32_t)(encoder->low + (current_range * cumulative_low) / total);

    for (;;) {
        if ((uint64_t)encoder->high < HALF_RANGE) {
            if (arithmetic_encoder_shift(encoder, 0) < 0) {
                return -1;
            }
        } else if ((uint64_t)encoder->low >= HALF_RANGE) {
            if (arithmetic_encoder_shift(encoder, 1) < 0) {
                return -1;
            }
            encoder->low -= (uint32_t)HALF_RANGE;
            encoder->high -= (uint32_t)HALF_RANGE;
        } else if ((uint64_t)encoder->low >= QUARTER_RANGE && (uint64_t)encoder->high < THREE_QUARTER_RANGE) {
            encoder->pending_bits += 1;
            encoder->low -= (uint32_t)QUARTER_RANGE;
            encoder->high -= (uint32_t)QUARTER_RANGE;
        } else {
            break;
        }

        encoder->low = (uint32_t)(((uint64_t)encoder->low << 1) & MASK);
        encoder->high = (uint32_t)((((uint64_t)encoder->high << 1) & MASK) | 1U);
    }

    return 0;
}


static int
arithmetic_encoder_finish(ArithmeticEncoder *encoder)
{
    encoder->pending_bits += 1;

    if ((uint64_t)encoder->low < QUARTER_RANGE) {
        if (arithmetic_encoder_shift(encoder, 0) < 0) {
            return -1;
        }
    } else {
        if (arithmetic_encoder_shift(encoder, 1) < 0) {
            return -1;
        }
    }

    return bit_writer_finish(encoder->output);
}


static void
arithmetic_decoder_init(ArithmeticDecoder *decoder, BitReader *reader)
{
    decoder->input = reader;
    decoder->low = 0;
    decoder->high = (uint32_t)MASK;
    decoder->code = 0;

    for (int index = 0; index < STATE_BITS; index++) {
        decoder->code = (uint32_t)((((uint64_t)decoder->code << 1) & MASK) | (uint32_t)bit_reader_read(reader));
    }
}


static uint32_t
arithmetic_decoder_get_target(ArithmeticDecoder *decoder, uint32_t total)
{
    uint64_t current_range = (uint64_t)decoder->high - decoder->low + 1U;
    uint64_t offset = (uint64_t)decoder->code - decoder->low;
    return (uint32_t)((((offset + 1U) * total) - 1U) / current_range);
}


static void
arithmetic_decoder_update(ArithmeticDecoder *decoder, uint32_t cumulative_low, uint32_t cumulative_high, uint32_t total)
{
    uint64_t current_range = (uint64_t)decoder->high - decoder->low + 1U;

    decoder->high = (uint32_t)(decoder->low + (current_range * cumulative_high) / total - 1U);
    decoder->low = (uint32_t)(decoder->low + (current_range * cumulative_low) / total);

    for (;;) {
        if ((uint64_t)decoder->high < HALF_RANGE) {
            /* no-op */
        } else if ((uint64_t)decoder->low >= HALF_RANGE) {
            decoder->low -= (uint32_t)HALF_RANGE;
            decoder->high -= (uint32_t)HALF_RANGE;
            decoder->code -= (uint32_t)HALF_RANGE;
        } else if ((uint64_t)decoder->low >= QUARTER_RANGE && (uint64_t)decoder->high < THREE_QUARTER_RANGE) {
            decoder->low -= (uint32_t)QUARTER_RANGE;
            decoder->high -= (uint32_t)QUARTER_RANGE;
            decoder->code -= (uint32_t)QUARTER_RANGE;
        } else {
            break;
        }

        decoder->low = (uint32_t)(((uint64_t)decoder->low << 1) & MASK);
        decoder->high = (uint32_t)((((uint64_t)decoder->high << 1) & MASK) | 1U);
        decoder->code = (uint32_t)((((uint64_t)decoder->code << 1) & MASK) | (uint32_t)bit_reader_read(decoder->input));
    }
}


static int
model_init(PPMModel *model, int order)
{
    memset(model, 0, sizeof(*model));
    model->order = order;
    model->root = node_arena_new_node(&model->arena);
    if (model->root == NULL) {
        return -1;
    }

    model->contexts = PyMem_Malloc((size_t)(order + 1) * sizeof(ContextNode *));
    model->next_contexts = PyMem_Malloc((size_t)(order + 1) * sizeof(ContextNode *));
    if (model->contexts == NULL || model->next_contexts == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    model->contexts[0] = model->root;
    model->context_len = 1;
    return 0;
}


static void
model_free(PPMModel *model)
{
    PyMem_Free(model->contexts);
    PyMem_Free(model->next_contexts);
    model->contexts = NULL;
    model->next_contexts = NULL;
    node_arena_free(&model->arena);
}


static int
model_encode_order_minus_one(int symbol, const uint64_t excluded_mask[MASK_WORDS], ArithmeticEncoder *encoder)
{
    uint32_t remaining = (uint32_t)(ALPHABET_SIZE - mask_bit_count(excluded_mask));
    uint32_t excluded_before_symbol = (uint32_t)mask_count_before_symbol(excluded_mask, symbol);
    uint32_t index = (uint32_t)symbol - excluded_before_symbol;
    return arithmetic_encoder_update(encoder, index, index + 1U, remaining);
}


static int
model_decode_order_minus_one(const uint64_t excluded_mask[MASK_WORDS], ArithmeticDecoder *decoder, int *symbol_out)
{
    uint32_t remaining = (uint32_t)(ALPHABET_SIZE - mask_bit_count(excluded_mask));
    uint32_t index = arithmetic_decoder_get_target(decoder, remaining);

    arithmetic_decoder_update(decoder, index, index + 1U, remaining);

    for (int symbol = 0; symbol < ALPHABET_SIZE; symbol++) {
        if (mask_contains(excluded_mask, symbol)) {
            continue;
        }
        if (index == 0) {
            *symbol_out = symbol;
            return 0;
        }
        index -= 1;
    }

    PyErr_SetString(PyExc_RuntimeError, "order -1 decoding failed");
    return -1;
}


static int
model_encode_symbol(PPMModel *model, int symbol, ArithmeticEncoder *encoder)
{
    uint64_t excluded_mask[MASK_WORDS] = {0, 0, 0, 0, 0};

    for (int ctx_index = model->context_len - 1; ctx_index >= 0; ctx_index--) {
        ContextNode *node = model->contexts[ctx_index];
        uint32_t visible_total = 0;
        uint32_t cumulative = 0;
        uint32_t symbol_low = 0;
        uint32_t symbol_count = 0;
        int found = 0;

        if (node->count_len == 0) {
            continue;
        }

        for (uint16_t index = 0; index < node->count_len; index++) {
            int candidate = node->counts[index].symbol;
            uint32_t count = node->counts[index].count;

            if (mask_contains(excluded_mask, candidate)) {
                continue;
            }

            if (candidate == symbol) {
                found = 1;
                symbol_low = cumulative;
                symbol_count = count;
            }

            cumulative += count;
            visible_total += count;
        }

        if (visible_total == 0) {
            mask_or(excluded_mask, node->symbol_mask);
            continue;
        }

        if (found) {
            return arithmetic_encoder_update(encoder, symbol_low, symbol_low + symbol_count, visible_total + 1U);
        }

        if (arithmetic_encoder_update(encoder, visible_total, visible_total + 1U, visible_total + 1U) < 0) {
            return -1;
        }
        mask_or(excluded_mask, node->symbol_mask);
    }

    return model_encode_order_minus_one(symbol, excluded_mask, encoder);
}


static int
model_decode_symbol(PPMModel *model, ArithmeticDecoder *decoder, int *symbol_out)
{
    uint64_t excluded_mask[MASK_WORDS] = {0, 0, 0, 0, 0};

    for (int ctx_index = model->context_len - 1; ctx_index >= 0; ctx_index--) {
        ContextNode *node = model->contexts[ctx_index];
        uint32_t visible_total = 0;

        if (node->count_len == 0) {
            continue;
        }

        for (uint16_t index = 0; index < node->count_len; index++) {
            int symbol = node->counts[index].symbol;
            if (mask_contains(excluded_mask, symbol)) {
                continue;
            }
            visible_total += node->counts[index].count;
        }

        if (visible_total == 0) {
            mask_or(excluded_mask, node->symbol_mask);
            continue;
        }

        uint32_t total = visible_total + 1U;
        uint32_t target = arithmetic_decoder_get_target(decoder, total);

        if (target == visible_total) {
            arithmetic_decoder_update(decoder, visible_total, visible_total + 1U, total);
            mask_or(excluded_mask, node->symbol_mask);
            continue;
        }

        uint32_t cumulative = 0;
        for (uint16_t index = 0; index < node->count_len; index++) {
            int symbol = node->counts[index].symbol;
            uint32_t count = node->counts[index].count;

            if (mask_contains(excluded_mask, symbol)) {
                continue;
            }

            if (target < cumulative + count) {
                arithmetic_decoder_update(decoder, cumulative, cumulative + count, total);
                *symbol_out = symbol;
                return 0;
            }
            cumulative += count;
        }
    }

    return model_decode_order_minus_one(excluded_mask, decoder, symbol_out);
}


static int
model_update(PPMModel *model, int symbol)
{
    for (int index = 0; index < model->context_len; index++) {
        if (context_node_increment(model->contexts[index], symbol) < 0) {
            return -1;
        }
    }

    if (model->order == 0 || symbol == EOF_SYMBOL) {
        return 0;
    }

    int limit = model->context_len < model->order ? model->context_len : model->order;
    int next_len = 1;
    model->next_contexts[0] = model->root;

    for (int index = 0; index < limit; index++) {
        ContextNode *child = context_node_get_or_add_child(model->contexts[index], symbol, &model->arena);
        if (child == NULL) {
            return -1;
        }
        model->next_contexts[next_len++] = child;
    }

    ContextNode **swap = model->contexts;
    model->contexts = model->next_contexts;
    model->next_contexts = swap;
    model->context_len = next_len;
    return 0;
}


static Py_ssize_t
varint_size(Py_ssize_t value)
{
    Py_ssize_t count = 1;
    while (value >= 0x80) {
        value >>= 7;
        count += 1;
    }
    return count;
}


static int
parse_fasta_lines_native(const uint8_t *data, Py_ssize_t len, int require_initial_header, FastaLineArray *lines)
{
    Py_ssize_t offset = 0;
    int saw_header = require_initial_header ? 0 : 1;

    if (len == 0) {
        PyErr_SetString(PyExc_ValueError, "empty input is not FASTA");
        return -1;
    }

    memset(lines, 0, sizeof(*lines));

    while (offset < len) {
        Py_ssize_t start = offset;
        Py_ssize_t end = len;
        uint8_t newline = NEWLINE_NONE;
        FastaLineView line;

        while (offset < len) {
            uint8_t current = data[offset];
            if (current == '\n') {
                end = offset;
                newline = NEWLINE_LF;
                offset += 1;
                break;
            }
            if (current == '\r') {
                end = offset;
                if (offset + 1 < len && data[offset + 1] == '\n') {
                    newline = NEWLINE_CRLF;
                    offset += 2;
                } else {
                    newline = NEWLINE_CR;
                    offset += 1;
                }
                break;
            }
            offset += 1;
        }

        line.is_header = data[start] == '>';
        if (!saw_header) {
            if (!line.is_header) {
                PyErr_SetString(PyExc_ValueError, "FASTA input must begin with a header line");
                fasta_line_array_free(lines);
                return -1;
            }
            saw_header = 1;
        }

        line.body = data + start + (line.is_header ? 1 : 0);
        line.body_len = end - start - (line.is_header ? 1 : 0);
        line.newline = newline;
        if (fasta_line_array_append(lines, line) < 0) {
            fasta_line_array_free(lines);
            return -1;
        }
    }

    return 0;
}


static void
build_sequence_alphabet(
    const FastaLineArray *lines,
    uint8_t alphabet[256],
    Py_ssize_t *alphabet_len,
    int16_t symbol_map[256]
)
{
    uint8_t present[256] = {0};

    for (int index = 0; index < 256; index++) {
        symbol_map[index] = -1;
    }

    for (Py_ssize_t line_index = 0; line_index < lines->len; line_index++) {
        const FastaLineView *line = &lines->items[line_index];
        if (line->is_header) {
            continue;
        }
        for (Py_ssize_t offset = 0; offset < line->body_len; offset++) {
            present[line->body[offset]] = 1;
        }
    }

    *alphabet_len = 0;
    for (int byte = 0; byte < 256; byte++) {
        if (present[byte]) {
            alphabet[*alphabet_len] = (uint8_t)byte;
            symbol_map[byte] = (int16_t)(*alphabet_len);
            *alphabet_len += 1;
        }
    }
}


static void
build_complement_map(
    const uint8_t *alphabet,
    Py_ssize_t alphabet_len,
    const int16_t symbol_map[256],
    uint8_t complement_map[256]
)
{
    for (Py_ssize_t index = 0; index < alphabet_len; index++) {
        uint8_t complement = complement_byte(alphabet[index]);
        int16_t mapped = symbol_map[complement];
        if (mapped >= 0) {
            complement_map[index] = (uint8_t)mapped;
        } else {
            complement_map[index] = (uint8_t)index;
        }
    }
}


static int
append_header_line(ByteBuffer *payload, const FastaLineView *line)
{
    if (byte_buffer_append_byte(payload, HEADER_TAG) < 0) {
        return -1;
    }
    if (append_varint(payload, line->body_len) < 0) {
        return -1;
    }
    if (byte_buffer_append_bytes(payload, line->body, line->body_len) < 0) {
        return -1;
    }
    return byte_buffer_append_byte(payload, line->newline);
}


static int
append_raw_sequence_line(
    ByteBuffer *payload,
    const FastaLineView *line,
    const int16_t symbol_map[256],
    Py_ssize_t alphabet_len,
    int allow_packed
)
{
    int packed = allow_packed && alphabet_len <= 4;

    if (byte_buffer_append_byte(payload, packed ? SEQUENCE_PACKED_TAG : SEQUENCE_TAG) < 0) {
        return -1;
    }
    if (append_varint(payload, line->body_len) < 0) {
        return -1;
    }
    if (packed) {
        ByteBuffer packed_body;
        byte_buffer_init(&packed_body);
        if (byte_buffer_reserve(&packed_body, packed_symbol_size(line->body_len)) < 0) {
            byte_buffer_free(&packed_body);
            return -1;
        }
        for (Py_ssize_t offset = 0; offset < line->body_len; offset++) {
            if (byte_buffer_append_byte(&packed_body, (uint8_t)symbol_map[line->body[offset]]) < 0) {
                byte_buffer_free(&packed_body);
                return -1;
            }
        }
        {
            ByteBuffer compressed_body;
            byte_buffer_init(&compressed_body);
            if (append_packed_symbols(&compressed_body, packed_body.data, packed_body.len) < 0) {
                byte_buffer_free(&packed_body);
                byte_buffer_free(&compressed_body);
                return -1;
            }
            byte_buffer_free(&packed_body);
            if (byte_buffer_append_bytes(payload, compressed_body.data, compressed_body.len) < 0) {
                byte_buffer_free(&compressed_body);
                return -1;
            }
            byte_buffer_free(&compressed_body);
        }
    } else {
        if (byte_buffer_reserve(payload, payload->len + line->body_len + 1) < 0) {
            return -1;
        }
        for (Py_ssize_t offset = 0; offset < line->body_len; offset++) {
            payload->data[payload->len++] = (uint8_t)symbol_map[line->body[offset]];
        }
    }
    return byte_buffer_append_byte(payload, line->newline);
}


static Py_ssize_t
sequence_line_size(Py_ssize_t body_len, Py_ssize_t alphabet_len, int allow_packed)
{
    return 1 + varint_size(body_len) + ((allow_packed && alphabet_len <= 4) ? packed_symbol_size(body_len) : body_len) + 1;
}


static int
append_group_packed_payload(ByteBuffer *payload, const FastaLineView *lines, Py_ssize_t group_start, Py_ssize_t group_count, const ByteBuffer *group_sequence)
{
    if (byte_buffer_append_byte(payload, SEQUENCE_GROUP_PACKED_TAG) < 0 || append_varint(payload, group_count) < 0) {
        return -1;
    }
    for (Py_ssize_t line_index = 0; line_index < group_count; line_index++) {
        if (append_varint(payload, lines[group_start + line_index].body_len) < 0 ||
            byte_buffer_append_byte(payload, lines[group_start + line_index].newline) < 0) {
            return -1;
        }
    }
    return append_packed_symbols(payload, group_sequence->data, group_sequence->len);
}


static Py_ssize_t
sequence_group_packed_size(const FastaLineView *lines, Py_ssize_t group_start, Py_ssize_t group_count)
{
    Py_ssize_t total_symbols = 0;
    Py_ssize_t meta = 0;

    for (Py_ssize_t index = 0; index < group_count; index++) {
        total_symbols += lines[group_start + index].body_len;
        meta += varint_size(lines[group_start + index].body_len) + 1;
    }
    return 1 + varint_size(group_count) + meta + packed_symbol_size(total_symbols);
}


static int
build_group_sequence(
    const FastaLineView *lines,
    Py_ssize_t line_count,
    const int16_t symbol_map[256],
    ByteBuffer *sequence
)
{
    Py_ssize_t total = 0;
    byte_buffer_init(sequence);

    for (Py_ssize_t line_index = 0; line_index < line_count; line_index++) {
        total += lines[line_index].body_len;
    }

    if (byte_buffer_reserve(sequence, total) < 0) {
        byte_buffer_free(sequence);
        return -1;
    }

    for (Py_ssize_t line_index = 0; line_index < line_count; line_index++) {
        const FastaLineView *line = &lines[line_index];
        for (Py_ssize_t offset = 0; offset < line->body_len; offset++) {
            sequence->data[sequence->len++] = (uint8_t)symbol_map[line->body[offset]];
        }
    }

    return 0;
}


static uint64_t
forward_kmer_key(const uint8_t *data)
{
    uint64_t key = 0;
    for (int index = 0; index < REPEAT_KMER_SIZE; index++) {
        key = (key << 8) | data[index];
    }
    return key;
}


static uint64_t
reverse_kmer_key(const uint8_t *data, const uint8_t complement_map[256], Py_ssize_t end)
{
    uint64_t key = 0;
    for (int index = 0; index < REPEAT_KMER_SIZE; index++) {
        key = (key << 8) | complement_map[data[end - index]];
    }
    return key;
}


static uint8_t
read_reference_sequence_byte(
    const uint8_t *history,
    Py_ssize_t history_len,
    const uint8_t *sequence,
    Py_ssize_t index
)
{
    if (index < history_len) {
        return history[index];
    }
    return sequence[index - history_len];
}


static uint8_t
read_decoded_reference_byte(const uint8_t *history, Py_ssize_t history_len, const ByteBuffer *decoded, Py_ssize_t index)
{
    if (index < history_len) {
        return history[index];
    }
    return decoded->data[index - history_len];
}


static int
index_history_sequence(
    const uint8_t *sequence,
    Py_ssize_t sequence_len,
    ByteBuffer *history,
    KmerIndex *forward_index,
    KmerIndex *reverse_index,
    const uint8_t complement_map[256]
)
{
    Py_ssize_t start = history->len;

    if (byte_buffer_append_bytes(history, sequence, sequence_len) < 0) {
        return -1;
    }
    if (history->len < REPEAT_KMER_SIZE) {
        return 0;
    }

    for (Py_ssize_t end = start > (REPEAT_KMER_SIZE - 1) ? start : (REPEAT_KMER_SIZE - 1); end < history->len; end++) {
        Py_ssize_t window_start = end - REPEAT_KMER_SIZE + 1;
        if (kmer_index_add(forward_index, forward_kmer_key(history->data + window_start), window_start) < 0) {
            return -1;
        }
        if (kmer_index_add(reverse_index, reverse_kmer_key(history->data, complement_map, end), end) < 0) {
            return -1;
        }
    }

    return 0;
}


static int
index_local_prefix(
    const uint8_t *sequence,
    Py_ssize_t indexed_prefix,
    Py_ssize_t prefix_length,
    Py_ssize_t history_len,
    KmerIndex *local_forward,
    KmerIndex *local_reverse,
    const uint8_t complement_map[256]
)
{
    if (prefix_length < REPEAT_KMER_SIZE) {
        return 0;
    }

    for (Py_ssize_t end = indexed_prefix > (REPEAT_KMER_SIZE - 1) ? indexed_prefix : (REPEAT_KMER_SIZE - 1); end < prefix_length; end++) {
        Py_ssize_t window_start = end - REPEAT_KMER_SIZE + 1;
        if (kmer_index_add(local_forward, forward_kmer_key(sequence + window_start), history_len + window_start) < 0) {
            return -1;
        }
        if (kmer_index_add(local_reverse, reverse_kmer_key(sequence, complement_map, end), history_len + end) < 0) {
            return -1;
        }
    }

    return 0;
}


static Py_ssize_t
forward_repeat_length(
    const uint8_t *sequence,
    Py_ssize_t sequence_len,
    Py_ssize_t position,
    Py_ssize_t source_start,
    const uint8_t *history,
    Py_ssize_t history_len
)
{
    Py_ssize_t length = REPEAT_KMER_SIZE;
    while (position + length < sequence_len) {
        if (read_reference_sequence_byte(history, history_len, sequence, source_start + length) != sequence[position + length]) {
            break;
        }
        length += 1;
    }
    return length;
}


static Py_ssize_t
reverse_repeat_length(
    const uint8_t *sequence,
    Py_ssize_t sequence_len,
    Py_ssize_t position,
    Py_ssize_t source_end,
    const uint8_t *history,
    Py_ssize_t history_len,
    const uint8_t complement_map[256]
)
{
    Py_ssize_t length = REPEAT_KMER_SIZE;
    Py_ssize_t max_length = sequence_len - position;
    if (source_end + 1 < max_length) {
        max_length = source_end + 1;
    }
    while (length < max_length) {
        uint8_t base = read_reference_sequence_byte(history, history_len, sequence, source_end - length);
        if (complement_map[base] != sequence[position + length]) {
            break;
        }
        length += 1;
    }
    return length;
}


static int
encode_literal_run(ByteBuffer *tokens, const uint8_t *sequence, Py_ssize_t start, Py_ssize_t end)
{
    if (start >= end) {
        return 0;
    }
    if (byte_buffer_append_byte(tokens, LITERAL_RUN_TAG) < 0) {
        return -1;
    }
    if (append_varint(tokens, end - start) < 0) {
        return -1;
    }
    return byte_buffer_append_bytes(tokens, sequence + start, end - start);
}


static int
encode_repeat_body_native(
    const uint8_t *sequence,
    Py_ssize_t sequence_len,
    const ByteBuffer *history,
    const KmerIndex *forward_index,
    const KmerIndex *reverse_index,
    const uint8_t complement_map[256],
    ByteBuffer *tokens,
    int *found_match_out
)
{
    KmerIndex local_forward = {0};
    KmerIndex local_reverse = {0};
    Py_ssize_t literal_start = 0;
    Py_ssize_t position = 0;
    Py_ssize_t indexed_prefix = 0;

    *found_match_out = 0;
    byte_buffer_init(tokens);

    if (sequence_len < REPEAT_MIN_MATCH) {
        return 0;
    }

    while (position < sequence_len) {
        uint64_t key = 0;
        Py_ssize_t current_reference_length = 0;
        int best_token = -1;
        Py_ssize_t best_length = 0;
        Py_ssize_t best_distance = 0;
        const KmerIndex *candidate_indexes[2] = {forward_index, &local_forward};
        const KmerIndex *reverse_candidate_indexes[2] = {reverse_index, &local_reverse};

        if (index_local_prefix(sequence, indexed_prefix, position, history->len, &local_forward, &local_reverse, complement_map) < 0) {
            goto error;
        }
        indexed_prefix = position;

        if (position + REPEAT_KMER_SIZE > sequence_len) {
            position += 1;
            continue;
        }

        key = forward_kmer_key(sequence + position);
        current_reference_length = history->len + position;

        for (int list_index = 0; list_index < 2; list_index++) {
            KmerEntry *entry = kmer_index_lookup_entry(candidate_indexes[list_index], key);
            if (entry == NULL) {
                continue;
            }
            for (int candidate = 0; candidate < entry->len; candidate++) {
                int ring_index = (entry->start + entry->len - 1 - candidate + REPEAT_MAX_CANDIDATES) % REPEAT_MAX_CANDIDATES;
                Py_ssize_t source_start = entry->positions[ring_index];
                Py_ssize_t length = forward_repeat_length(sequence, sequence_len, position, source_start, history->data, history->len);
                if (length > best_length) {
                    best_token = MATCH_RUN_TAG;
                    best_length = length;
                    best_distance = current_reference_length - source_start;
                }
            }
        }

        for (int list_index = 0; list_index < 2; list_index++) {
            KmerEntry *entry = kmer_index_lookup_entry(reverse_candidate_indexes[list_index], key);
            if (entry == NULL) {
                continue;
            }
            for (int candidate = 0; candidate < entry->len; candidate++) {
                int ring_index = (entry->start + entry->len - 1 - candidate + REPEAT_MAX_CANDIDATES) % REPEAT_MAX_CANDIDATES;
                Py_ssize_t source_end = entry->positions[ring_index];
                Py_ssize_t length = reverse_repeat_length(sequence, sequence_len, position, source_end, history->data, history->len, complement_map);
                if (length > best_length) {
                    best_token = REVERSE_COMPLEMENT_RUN_TAG;
                    best_length = length;
                    best_distance = current_reference_length - 1 - source_end;
                }
            }
        }

        if (best_length < REPEAT_MIN_MATCH) {
            position += 1;
            continue;
        }

        if (encode_literal_run(tokens, sequence, literal_start, position) < 0) {
            goto error;
        }
        if (byte_buffer_append_byte(tokens, (uint8_t)best_token) < 0) {
            goto error;
        }
        if (append_varint(tokens, best_length) < 0 || append_varint(tokens, best_distance) < 0) {
            goto error;
        }
        position += best_length;
        literal_start = position;
        *found_match_out = 1;
    }

    if (*found_match_out && encode_literal_run(tokens, sequence, literal_start, sequence_len) < 0) {
        goto error;
    }

    kmer_index_free(&local_forward);
    kmer_index_free(&local_reverse);
    return 0;

error:
    kmer_index_free(&local_forward);
    kmer_index_free(&local_reverse);
    byte_buffer_free(tokens);
    return -1;
}


static int
append_decoded_sequence(ByteBuffer *output, const uint8_t *alphabet, Py_ssize_t alphabet_len, const uint8_t *body, Py_ssize_t body_len)
{
    if (byte_buffer_reserve(output, output->len + body_len) < 0) {
        return -1;
    }
    for (Py_ssize_t index = 0; index < body_len; index++) {
        if (body[index] >= alphabet_len) {
            PyErr_SetString(PyExc_ValueError, "FASTA payload references an unknown alphabet symbol");
            return -1;
        }
        output->data[output->len++] = alphabet[body[index]];
    }
    return 0;
}


static int
append_newline_code(ByteBuffer *output, uint8_t newline)
{
    if (newline == NEWLINE_NONE) {
        return 0;
    }
    if (newline == NEWLINE_LF) {
        return byte_buffer_append_byte(output, '\n');
    }
    if (newline == NEWLINE_CRLF) {
        return byte_buffer_append_bytes(output, (const uint8_t *)"\r\n", 2);
    }
    if (newline == NEWLINE_CR) {
        return byte_buffer_append_byte(output, '\r');
    }
    PyErr_SetString(PyExc_ValueError, "unknown newline code");
    return -1;
}


static Py_ssize_t
sequence_group_size(const FastaLineView *lines, Py_ssize_t group_start, Py_ssize_t group_count, Py_ssize_t encoded_body_len)
{
    Py_ssize_t meta = 0;
    for (Py_ssize_t index = 0; index < group_count; index++) {
        meta += varint_size(lines[group_start + index].body_len) + 1;
    }
    return 1 + varint_size(group_count) + meta + varint_size(encoded_body_len) + encoded_body_len;
}


static int
decode_repeat_body_native(
    const uint8_t *encoded,
    Py_ssize_t encoded_len,
    Py_ssize_t expected_len,
    const ByteBuffer *history,
    const uint8_t complement_map[256],
    ByteBuffer *decoded
)
{
    Py_ssize_t offset = 0;
    byte_buffer_init(decoded);

    while (offset < encoded_len) {
        Py_ssize_t run_length = 0;
        uint8_t token = encoded[offset++];
        if (read_varint(encoded, encoded_len, &offset, &run_length) < 0) {
            goto error;
        }

        if (token == LITERAL_RUN_TAG) {
            if (offset + run_length > encoded_len) {
                PyErr_SetString(PyExc_ValueError, "truncated FASTA repeat literal");
                goto error;
            }
            if (byte_buffer_append_bytes(decoded, encoded + offset, run_length) < 0) {
                goto error;
            }
            offset += run_length;
            continue;
        }

        Py_ssize_t distance = 0;
        if (read_varint(encoded, encoded_len, &offset, &distance) < 0) {
            goto error;
        }

        if (token == MATCH_RUN_TAG) {
            Py_ssize_t source_start = history->len + decoded->len - distance;
            if (source_start < 0) {
                PyErr_SetString(PyExc_ValueError, "invalid FASTA repeat match");
                goto error;
            }
            for (Py_ssize_t copied = 0; copied < run_length; copied++) {
                if (byte_buffer_append_byte(decoded, read_decoded_reference_byte(history->data, history->len, decoded, source_start + copied)) < 0) {
                    goto error;
                }
            }
            continue;
        }

        if (token == REVERSE_COMPLEMENT_RUN_TAG) {
            Py_ssize_t source_end = history->len + decoded->len - 1 - distance;
            if (source_end < 0 || source_end - run_length + 1 < 0) {
                PyErr_SetString(PyExc_ValueError, "invalid FASTA reverse-complement match");
                goto error;
            }
            for (Py_ssize_t source = source_end; source > source_end - run_length; source--) {
                if (byte_buffer_append_byte(decoded, complement_map[read_decoded_reference_byte(history->data, history->len, decoded, source)]) < 0) {
                    goto error;
                }
            }
            continue;
        }

        PyErr_SetString(PyExc_ValueError, "unknown FASTA repeat token");
        goto error;
    }

    if (decoded->len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "decoded FASTA repeat length does not match metadata");
        goto error;
    }

    return 0;

error:
    byte_buffer_free(decoded);
    return -1;
}


static PyObject *
py_transform_fasta(PyObject *Py_UNUSED(module), PyObject *args)
{
    Py_buffer input = {0};
    int repeat_mode = 0;
    int require_initial_header = 1;
    int packed_mode = 0;
    FastaLineArray lines = {0};
    ByteBuffer payload;
    uint8_t alphabet[256];
    Py_ssize_t alphabet_len = 0;
    int16_t symbol_map[256];
    uint8_t complement_map[256] = {0};
    ByteBuffer history;
    KmerIndex forward_index = {0};
    KmerIndex reverse_index = {0};
    PyObject *payload_obj = NULL;
    PyObject *alphabet_obj = NULL;
    PyObject *result = NULL;

    byte_buffer_init(&payload);
    byte_buffer_init(&history);

    if (!PyArg_ParseTuple(args, "y*ppp", &input, &repeat_mode, &require_initial_header, &packed_mode)) {
        return NULL;
    }

    if (parse_fasta_lines_native((const uint8_t *)input.buf, input.len, require_initial_header, &lines) < 0) {
        goto done;
    }

    build_sequence_alphabet(&lines, alphabet, &alphabet_len, symbol_map);
    build_complement_map(alphabet, alphabet_len, symbol_map, complement_map);

    for (Py_ssize_t offset = 0; offset < lines.len;) {
        FastaLineView *line = &lines.items[offset];
        if (line->is_header) {
            if (append_header_line(&payload, line) < 0) {
                goto done;
            }
            offset += 1;
            continue;
        }

        if (!repeat_mode) {
            if (append_raw_sequence_line(&payload, line, symbol_map, alphabet_len, packed_mode) < 0) {
                goto done;
            }
            offset += 1;
            continue;
        }

        Py_ssize_t group_start = offset;
        Py_ssize_t group_count = 0;
        Py_ssize_t raw_size = 0;
        ByteBuffer group_sequence;
        ByteBuffer encoded_body;
        int found_match = 0;
        int use_group = 0;
        int use_packed_group = 0;

        while (offset < lines.len && !lines.items[offset].is_header) {
            raw_size += sequence_line_size(lines.items[offset].body_len, alphabet_len, 1);
            offset += 1;
            group_count += 1;
        }

        if (build_group_sequence(&lines.items[group_start], group_count, symbol_map, &group_sequence) < 0) {
            goto done;
        }
        if (encode_repeat_body_native(group_sequence.data, group_sequence.len, &history, &forward_index, &reverse_index, complement_map, &encoded_body, &found_match) < 0) {
            byte_buffer_free(&group_sequence);
            goto done;
        }
        if (!found_match) {
            byte_buffer_free(&encoded_body);
            byte_buffer_init(&encoded_body);
            if (byte_buffer_append_byte(&encoded_body, LITERAL_RUN_TAG) < 0) {
                byte_buffer_free(&group_sequence);
                byte_buffer_free(&encoded_body);
                goto done;
            }
            if (append_varint(&encoded_body, group_sequence.len) < 0 || byte_buffer_append_bytes(&encoded_body, group_sequence.data, group_sequence.len) < 0) {
                byte_buffer_free(&group_sequence);
                byte_buffer_free(&encoded_body);
                goto done;
            }
        }

        use_group = sequence_group_size(lines.items, group_start, group_count, encoded_body.len) < raw_size;
        if (alphabet_len <= 4) {
            Py_ssize_t packed_group_size = sequence_group_packed_size(lines.items, group_start, group_count);
            if (packed_group_size < raw_size && packed_group_size <= sequence_group_size(lines.items, group_start, group_count, encoded_body.len)) {
                use_group = 0;
                use_packed_group = 1;
            }
        }

        if (use_group) {
            if (byte_buffer_append_byte(&payload, SEQUENCE_GROUP_TAG) < 0 || append_varint(&payload, group_count) < 0) {
                byte_buffer_free(&group_sequence);
                byte_buffer_free(&encoded_body);
                goto done;
            }
            for (Py_ssize_t line_index = 0; line_index < group_count; line_index++) {
                if (append_varint(&payload, lines.items[group_start + line_index].body_len) < 0 ||
                    byte_buffer_append_byte(&payload, lines.items[group_start + line_index].newline) < 0) {
                    byte_buffer_free(&group_sequence);
                    byte_buffer_free(&encoded_body);
                    goto done;
                }
            }
            if (append_varint(&payload, encoded_body.len) < 0 || byte_buffer_append_bytes(&payload, encoded_body.data, encoded_body.len) < 0) {
                byte_buffer_free(&group_sequence);
                byte_buffer_free(&encoded_body);
                goto done;
            }
        } else if (use_packed_group) {
            if (append_group_packed_payload(&payload, lines.items, group_start, group_count, &group_sequence) < 0) {
                byte_buffer_free(&group_sequence);
                byte_buffer_free(&encoded_body);
                goto done;
            }
        } else {
            for (Py_ssize_t line_index = 0; line_index < group_count; line_index++) {
                if (append_raw_sequence_line(&payload, &lines.items[group_start + line_index], symbol_map, alphabet_len, 1) < 0) {
                    byte_buffer_free(&group_sequence);
                    byte_buffer_free(&encoded_body);
                    goto done;
                }
            }
        }

        if (index_history_sequence(group_sequence.data, group_sequence.len, &history, &forward_index, &reverse_index, complement_map) < 0) {
            byte_buffer_free(&group_sequence);
            byte_buffer_free(&encoded_body);
            goto done;
        }

        byte_buffer_free(&group_sequence);
        byte_buffer_free(&encoded_body);
    }

    payload_obj = byte_buffer_to_bytes(&payload);
    alphabet_obj = PyBytes_FromStringAndSize((const char *)alphabet, alphabet_len);
    if (payload_obj == NULL || alphabet_obj == NULL) {
        goto done;
    }

    result = PyTuple_Pack(2, payload_obj, alphabet_obj);

done:
    Py_XDECREF(payload_obj);
    Py_XDECREF(alphabet_obj);
    PyBuffer_Release(&input);
    fasta_line_array_free(&lines);
    byte_buffer_free(&payload);
    byte_buffer_free(&history);
    kmer_index_free(&forward_index);
    kmer_index_free(&reverse_index);
    return result;
}


static PyObject *
py_restore_fasta(PyObject *Py_UNUSED(module), PyObject *args)
{
    Py_buffer payload = {0};
    Py_buffer alphabet_buffer = {0};
    ByteBuffer output;
    ByteBuffer history;
    uint8_t complement_map[256] = {0};
    PyObject *result = NULL;
    Py_ssize_t offset = 0;

    byte_buffer_init(&output);
    byte_buffer_init(&history);

    if (!PyArg_ParseTuple(args, "y*y*", &payload, &alphabet_buffer)) {
        return NULL;
    }

    {
        int16_t symbol_map[256];
        for (int index = 0; index < 256; index++) {
            symbol_map[index] = -1;
        }
        for (Py_ssize_t index = 0; index < alphabet_buffer.len; index++) {
            symbol_map[((const uint8_t *)alphabet_buffer.buf)[index]] = (int16_t)index;
        }
        build_complement_map((const uint8_t *)alphabet_buffer.buf, alphabet_buffer.len, symbol_map, complement_map);
    }

    while (offset < payload.len) {
        uint8_t tag = ((const uint8_t *)payload.buf)[offset++];
        Py_ssize_t length = 0;

        if (read_varint((const uint8_t *)payload.buf, payload.len, &offset, &length) < 0) {
            goto done;
        }

        if (tag == HEADER_TAG) {
            if (offset + length + 1 > payload.len) {
                PyErr_SetString(PyExc_ValueError, "truncated FASTA payload");
                goto done;
            }
            if (byte_buffer_append_byte(&output, '>') < 0 ||
                byte_buffer_append_bytes(&output, (const uint8_t *)payload.buf + offset, length) < 0) {
                goto done;
            }
            offset += length;
            {
                uint8_t newline = ((const uint8_t *)payload.buf)[offset++];
                if (append_newline_code(&output, newline) < 0) {
                    goto done;
                }
            }
            continue;
        }

        if (tag == SEQUENCE_TAG) {
            if (offset + length + 1 > payload.len) {
                PyErr_SetString(PyExc_ValueError, "truncated FASTA payload");
                goto done;
            }
            if (append_decoded_sequence(&output, (const uint8_t *)alphabet_buffer.buf, alphabet_buffer.len, (const uint8_t *)payload.buf + offset, length) < 0 ||
                byte_buffer_append_bytes(&history, (const uint8_t *)payload.buf + offset, length) < 0) {
                goto done;
            }
            offset += length;
            {
                uint8_t newline = ((const uint8_t *)payload.buf)[offset++];
                if (append_newline_code(&output, newline) < 0) {
                    goto done;
                }
            }
            continue;
        }

        if (tag == SEQUENCE_PACKED_TAG) {
            Py_ssize_t packed_len = packed_symbol_size(length);
            ByteBuffer unpacked;
            byte_buffer_init(&unpacked);
            if (offset + packed_len + 1 > payload.len) {
                PyErr_SetString(PyExc_ValueError, "truncated FASTA packed payload");
                goto done;
            }
            if (unpack_symbols(&unpacked, (const uint8_t *)payload.buf + offset, length) < 0) {
                goto done;
            }
            if (append_decoded_sequence(&output, (const uint8_t *)alphabet_buffer.buf, alphabet_buffer.len, unpacked.data, unpacked.len) < 0 ||
                byte_buffer_append_bytes(&history, unpacked.data, unpacked.len) < 0) {
                byte_buffer_free(&unpacked);
                goto done;
            }
            byte_buffer_free(&unpacked);
            offset += packed_len;
            {
                uint8_t newline = ((const uint8_t *)payload.buf)[offset++];
                if (append_newline_code(&output, newline) < 0) {
                    goto done;
                }
            }
            continue;
        }

        if (tag == SEQUENCE_REPEAT_TAG) {
            Py_ssize_t encoded_length = 0;
            ByteBuffer decoded;
            if (read_varint((const uint8_t *)payload.buf, payload.len, &offset, &encoded_length) < 0) {
                goto done;
            }
            if (offset + encoded_length + 1 > payload.len) {
                PyErr_SetString(PyExc_ValueError, "truncated FASTA repeat payload");
                goto done;
            }
            if (decode_repeat_body_native((const uint8_t *)payload.buf + offset, encoded_length, length, &history, complement_map, &decoded) < 0) {
                goto done;
            }
            if (append_decoded_sequence(&output, (const uint8_t *)alphabet_buffer.buf, alphabet_buffer.len, decoded.data, decoded.len) < 0 ||
                byte_buffer_append_bytes(&history, decoded.data, decoded.len) < 0) {
                byte_buffer_free(&decoded);
                goto done;
            }
            byte_buffer_free(&decoded);
            offset += encoded_length;
            {
                uint8_t newline = ((const uint8_t *)payload.buf)[offset++];
                if (append_newline_code(&output, newline) < 0) {
                    goto done;
                }
            }
            continue;
        }

        if (tag == SEQUENCE_GROUP_TAG) {
            Py_ssize_t line_count = length;
            Py_ssize_t *line_lengths = NULL;
            uint8_t *newlines = NULL;
            Py_ssize_t total_length = 0;
            Py_ssize_t encoded_length = 0;
            ByteBuffer decoded;

            line_lengths = PyMem_Malloc((size_t)line_count * sizeof(Py_ssize_t));
            newlines = PyMem_Malloc((size_t)line_count * sizeof(uint8_t));
            if (line_lengths == NULL || newlines == NULL) {
                PyMem_Free(line_lengths);
                PyMem_Free(newlines);
                PyErr_NoMemory();
                goto done;
            }

            for (Py_ssize_t line_index = 0; line_index < line_count; line_index++) {
                if (read_varint((const uint8_t *)payload.buf, payload.len, &offset, &line_lengths[line_index]) < 0) {
                    PyMem_Free(line_lengths);
                    PyMem_Free(newlines);
                    goto done;
                }
                if (offset >= payload.len) {
                    PyMem_Free(line_lengths);
                    PyMem_Free(newlines);
                    PyErr_SetString(PyExc_ValueError, "truncated FASTA sequence-group metadata");
                    goto done;
                }
                newlines[line_index] = ((const uint8_t *)payload.buf)[offset++];
                total_length += line_lengths[line_index];
            }

            if (read_varint((const uint8_t *)payload.buf, payload.len, &offset, &encoded_length) < 0) {
                PyMem_Free(line_lengths);
                PyMem_Free(newlines);
                goto done;
            }
            if (offset + encoded_length > payload.len) {
                PyMem_Free(line_lengths);
                PyMem_Free(newlines);
                PyErr_SetString(PyExc_ValueError, "truncated FASTA sequence-group payload");
                goto done;
            }
            if (decode_repeat_body_native((const uint8_t *)payload.buf + offset, encoded_length, total_length, &history, complement_map, &decoded) < 0) {
                PyMem_Free(line_lengths);
                PyMem_Free(newlines);
                goto done;
            }
            offset += encoded_length;

            {
                Py_ssize_t decoded_offset = 0;
                for (Py_ssize_t line_index = 0; line_index < line_count; line_index++) {
                    if (append_decoded_sequence(&output, (const uint8_t *)alphabet_buffer.buf, alphabet_buffer.len, decoded.data + decoded_offset, line_lengths[line_index]) < 0) {
                        byte_buffer_free(&decoded);
                        PyMem_Free(line_lengths);
                        PyMem_Free(newlines);
                        goto done;
                    }
                    decoded_offset += line_lengths[line_index];
                    if (append_newline_code(&output, newlines[line_index]) < 0) {
                        byte_buffer_free(&decoded);
                        PyMem_Free(line_lengths);
                        PyMem_Free(newlines);
                        goto done;
                    }
                }
            }

            if (byte_buffer_append_bytes(&history, decoded.data, decoded.len) < 0) {
                byte_buffer_free(&decoded);
                PyMem_Free(line_lengths);
                PyMem_Free(newlines);
                goto done;
            }

            byte_buffer_free(&decoded);
            PyMem_Free(line_lengths);
            PyMem_Free(newlines);
            continue;
        }

        if (tag == SEQUENCE_GROUP_PACKED_TAG) {
            Py_ssize_t line_count = length;
            Py_ssize_t *line_lengths = NULL;
            uint8_t *newlines = NULL;
            Py_ssize_t total_length = 0;
            ByteBuffer unpacked;
            byte_buffer_init(&unpacked);

            line_lengths = PyMem_Malloc((size_t)line_count * sizeof(Py_ssize_t));
            newlines = PyMem_Malloc((size_t)line_count * sizeof(uint8_t));
            if (line_lengths == NULL || newlines == NULL) {
                PyMem_Free(line_lengths);
                PyMem_Free(newlines);
                PyErr_NoMemory();
                goto done;
            }

            for (Py_ssize_t line_index = 0; line_index < line_count; line_index++) {
                if (read_varint((const uint8_t *)payload.buf, payload.len, &offset, &line_lengths[line_index]) < 0) {
                    PyMem_Free(line_lengths);
                    PyMem_Free(newlines);
                    goto done;
                }
                if (offset >= payload.len) {
                    PyMem_Free(line_lengths);
                    PyMem_Free(newlines);
                    PyErr_SetString(PyExc_ValueError, "truncated FASTA sequence-group metadata");
                    goto done;
                }
                newlines[line_index] = ((const uint8_t *)payload.buf)[offset++];
                total_length += line_lengths[line_index];
            }

            if (offset + packed_symbol_size(total_length) > payload.len) {
                PyMem_Free(line_lengths);
                PyMem_Free(newlines);
                PyErr_SetString(PyExc_ValueError, "truncated FASTA packed sequence-group payload");
                goto done;
            }
            if (unpack_symbols(&unpacked, (const uint8_t *)payload.buf + offset, total_length) < 0) {
                PyMem_Free(line_lengths);
                PyMem_Free(newlines);
                goto done;
            }
            offset += packed_symbol_size(total_length);

            {
                Py_ssize_t unpacked_offset = 0;
                for (Py_ssize_t line_index = 0; line_index < line_count; line_index++) {
                    if (append_decoded_sequence(&output, (const uint8_t *)alphabet_buffer.buf, alphabet_buffer.len, unpacked.data + unpacked_offset, line_lengths[line_index]) < 0) {
                        byte_buffer_free(&unpacked);
                        PyMem_Free(line_lengths);
                        PyMem_Free(newlines);
                        goto done;
                    }
                    unpacked_offset += line_lengths[line_index];
                    if (append_newline_code(&output, newlines[line_index]) < 0) {
                        byte_buffer_free(&unpacked);
                        PyMem_Free(line_lengths);
                        PyMem_Free(newlines);
                        goto done;
                    }
                }
            }

            if (byte_buffer_append_bytes(&history, unpacked.data, unpacked.len) < 0) {
                byte_buffer_free(&unpacked);
                PyMem_Free(line_lengths);
                PyMem_Free(newlines);
                goto done;
            }

            byte_buffer_free(&unpacked);
            PyMem_Free(line_lengths);
            PyMem_Free(newlines);
            continue;
        }

        PyErr_SetString(PyExc_ValueError, "unknown FASTA payload tag");
        goto done;
    }

    result = byte_buffer_to_bytes(&output);

done:
    byte_buffer_free(&output);
    byte_buffer_free(&history);
    PyBuffer_Release(&payload);
    PyBuffer_Release(&alphabet_buffer);
    return result;
}


static int
compare_u64(const void *left_ptr, const void *right_ptr)
{
    uint64_t left = *(const uint64_t *)left_ptr;
    uint64_t right = *(const uint64_t *)right_ptr;
    if (left < right) {
        return -1;
    }
    if (left > right) {
        return 1;
    }
    return 0;
}


static PyObject *
py_build_kmer_signature(PyObject *Py_UNUSED(module), PyObject *args)
{
    static const uint8_t magic[] = {'K', 'M', 'S', '1'};
    Py_buffer input = {0};
    int kmer_size = 17;
    int sample_rate = 8;
    FastaLineArray lines = {0};
    UInt64Buffer kmers;
    ByteBuffer signature;
    uint64_t mask = 0;
    uint64_t forward = 0;
    uint64_t reverse = 0;
    uint64_t previous = 0;
    uint64_t shift = 0;
    Py_ssize_t valid = 0;
    PyObject *result = NULL;

    u64_buffer_init(&kmers);
    byte_buffer_init(&signature);

    if (!PyArg_ParseTuple(args, "y*ii", &input, &kmer_size, &sample_rate)) {
        return NULL;
    }

    if (kmer_size < 1 || kmer_size > 31) {
        PyErr_SetString(PyExc_ValueError, "kmer_size must be between 1 and 31");
        goto done;
    }
    if (sample_rate < 1) {
        PyErr_SetString(PyExc_ValueError, "sample_rate must be at least 1");
        goto done;
    }

    if (parse_fasta_lines_native((const uint8_t *)input.buf, input.len, 1, &lines) < 0) {
        goto done;
    }

    mask = (UINT64_C(1) << (2 * kmer_size)) - 1;
    shift = (uint64_t)(2 * (kmer_size - 1));

    for (Py_ssize_t line_index = 0; line_index < lines.len; line_index++) {
        const FastaLineView *line = &lines.items[line_index];
        if (line->is_header) {
            forward = 0;
            reverse = 0;
            valid = 0;
            continue;
        }

        for (Py_ssize_t offset = 0; offset < line->body_len; offset++) {
            int code = dna_base_code(line->body[offset]);
            uint64_t canonical = 0;

            if (code < 0) {
                forward = 0;
                reverse = 0;
                valid = 0;
                continue;
            }

            forward = ((forward << 2) | (uint64_t)code) & mask;
            reverse = (reverse >> 2) | ((uint64_t)(code ^ 0x03) << shift);
            if (valid < kmer_size) {
                valid += 1;
            }
            if (valid < kmer_size) {
                continue;
            }

            canonical = (forward < reverse) ? forward : reverse;
            if (sample_rate > 1 && (splitmix64(canonical) % (uint64_t)sample_rate) != 0) {
                continue;
            }
            if (u64_buffer_append(&kmers, canonical) < 0) {
                goto done;
            }
        }
    }

    if (kmers.len > 1) {
        qsort(kmers.data, (size_t)kmers.len, sizeof(uint64_t), compare_u64);
        {
            Py_ssize_t unique_len = 1;
            for (Py_ssize_t index = 1; index < kmers.len; index++) {
                if (kmers.data[index] != kmers.data[unique_len - 1]) {
                    kmers.data[unique_len++] = kmers.data[index];
                }
            }
            kmers.len = unique_len;
        }
    }

    if (byte_buffer_append_bytes(&signature, magic, 4) < 0 ||
        append_varint(&signature, kmer_size) < 0 ||
        append_varint(&signature, sample_rate) < 0 ||
        append_varint(&signature, kmers.len) < 0) {
        goto done;
    }

    for (Py_ssize_t index = 0; index < kmers.len; index++) {
        if (append_varint(&signature, (Py_ssize_t)(kmers.data[index] - previous)) < 0) {
            goto done;
        }
        previous = kmers.data[index];
    }

    result = byte_buffer_to_bytes(&signature);

done:
    fasta_line_array_free(&lines);
    u64_buffer_free(&kmers);
    byte_buffer_free(&signature);
    PyBuffer_Release(&input);
    return result;
}


static PyObject *
py_compress_payload(PyObject *Py_UNUSED(module), PyObject *args)
{
    Py_buffer input = {0};
    int order = 0;
    PPMModel model;
    BitWriter writer;
    ArithmeticEncoder encoder;
    PyObject *result = NULL;

    memset(&model, 0, sizeof(model));
    memset(&writer, 0, sizeof(writer));

    if (!PyArg_ParseTuple(args, "y*i", &input, &order)) {
        return NULL;
    }

    if (order < 0 || order > 16) {
        PyBuffer_Release(&input);
        PyErr_SetString(PyExc_ValueError, "order must be between 0 and 16");
        return NULL;
    }

    if (bit_writer_init(&writer, input.len / 2 + 16) < 0) {
        goto done;
    }
    if (model_init(&model, order) < 0) {
        goto done;
    }

    arithmetic_encoder_init(&encoder, &writer);

    const uint8_t *data = (const uint8_t *)input.buf;
    for (Py_ssize_t index = 0; index < input.len; index++) {
        if (model_encode_symbol(&model, data[index], &encoder) < 0) {
            goto done;
        }
        if (model_update(&model, data[index]) < 0) {
            goto done;
        }
    }

    if (model_encode_symbol(&model, EOF_SYMBOL, &encoder) < 0) {
        goto done;
    }
    if (arithmetic_encoder_finish(&encoder) < 0) {
        goto done;
    }

    result = PyBytes_FromStringAndSize((const char *)writer.data, writer.len);

done:
    PyBuffer_Release(&input);
    bit_writer_free(&writer);
    model_free(&model);
    return result;
}


static PyObject *
py_decompress_payload(PyObject *Py_UNUSED(module), PyObject *args)
{
    Py_buffer input = {0};
    int order = 0;
    PPMModel model;
    BitReader reader;
    ArithmeticDecoder decoder;
    uint8_t *output = NULL;
    Py_ssize_t output_len = 0;
    Py_ssize_t output_cap = 0;
    PyObject *result = NULL;

    memset(&model, 0, sizeof(model));

    if (!PyArg_ParseTuple(args, "y*i", &input, &order)) {
        return NULL;
    }

    if (order < 0 || order > 16) {
        PyBuffer_Release(&input);
        PyErr_SetString(PyExc_ValueError, "order must be between 0 and 16");
        return NULL;
    }

    if (model_init(&model, order) < 0) {
        goto done;
    }

    bit_reader_init(&reader, (const uint8_t *)input.buf, input.len);
    arithmetic_decoder_init(&decoder, &reader);

    for (;;) {
        int symbol = 0;

        if (model_decode_symbol(&model, &decoder, &symbol) < 0) {
            goto done;
        }

        if (symbol == EOF_SYMBOL) {
            break;
        }

        if (ensure_capacity((void **)&output, &output_cap, sizeof(uint8_t), output_len + 1) < 0) {
            goto done;
        }

        output[output_len++] = (uint8_t)symbol;
        if (model_update(&model, symbol) < 0) {
            goto done;
        }
    }

    result = PyBytes_FromStringAndSize((const char *)output, output_len);

done:
    PyBuffer_Release(&input);
    PyMem_Free(output);
    model_free(&model);
    return result;
}


static PyMethodDef module_methods[] = {
    {"compress_payload", py_compress_payload, METH_VARARGS, "Compress a payload with the native PPM coder."},
    {"decompress_payload", py_decompress_payload, METH_VARARGS, "Decompress a payload with the native PPM coder."},
    {"transform_fasta", py_transform_fasta, METH_VARARGS, "Transform FASTA bytes into a payload and alphabet."},
    {"restore_fasta", py_restore_fasta, METH_VARARGS, "Restore FASTA bytes from a transformed payload and alphabet."},
    {"build_kmer_signature", py_build_kmer_signature, METH_VARARGS, "Build a canonical FASTA k-mer signature."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_cppm",
    "Native PPM payload coder for ppmdc.",
    -1,
    module_methods,
};


PyMODINIT_FUNC
PyInit__cppm(void)
{
    return PyModule_Create(&module_def);
}
