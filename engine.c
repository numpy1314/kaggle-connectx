/* Native search engine for the standard 7-column, 6-row Connect Four board. */
#include <stdint.h>

#if defined(CX_CROSS_LINUX)
/* x86-64 Linux ABI, used when building without a Linux sysroot. */
struct timespec { long tv_sec; long tv_nsec; };
extern int clock_gettime(int clock_id, struct timespec *time);
#define CLOCK_MONOTONIC 1
#else
#include <time.h>
#endif

typedef uint64_t Bits;

enum { WIDTH = 7, HEIGHT = 6, STRIDE = 7, CELLS = 42 };
enum { MATE = 10000, INF = MATE + 1, HEURISTIC_LIMIT = 900 };
enum { EXACT = 1, LOWER = 2, UPPER = 3 };
enum { LOSS = -1, DRAW = 0, WIN = 1, UNKNOWN = 2 };
enum { UNTESTED = 0, IMPOSSIBLE = 1, SUCCESS = 2 };

#ifndef CX_PROOF_BITS
#define CX_PROOF_BITS 23
#endif
#ifndef CX_SEARCH_BITS
#define CX_SEARCH_BITS 20
#endif

#define EXPORT __attribute__((visibility("default")))
#define LOW_BITS(n) ((UINT64_C(1) << (n)) - 1)
#define COLUMN(c) (LOW_BITS(HEIGHT) << (STRIDE * (c)))
#define BOTTOM UINT64_C(0x40810204081)
#define BOARD (BOTTOM * LOW_BITS(HEIGHT))
#define EVEN_ROWS (BOTTOM * 21)
#define ODD_ROWS (BOARD ^ EVEN_ROWS)
#define KEY_BITS (WIDTH * STRIDE)
#define KEY_MASK LOW_BITS(KEY_BITS)

static const int center_order[WIDTH] = {3, 2, 4, 1, 5, 0, 6};
static const Bits columns[WIDTH] = {
    COLUMN(0), COLUMN(1), COLUMN(2), COLUMN(3), COLUMN(4), COLUMN(5), COLUMN(6)
};

typedef struct {
    Bits key;
    int16_t score;
    uint8_t depth;
    int8_t column;
    uint8_t bound;
} SearchEntry;

typedef struct {
    double deadline;
    uint64_t nodes;
    int stopped;
} Search;

typedef struct {
    Bits move;
    int column;
    int priority;
} Move;

static Bits proof_table[UINT64_C(1) << CX_PROOF_BITS];
static SearchEntry search_table[UINT64_C(1) << CX_SEARCH_BITS];

static double monotonic_seconds(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) {
        return 0.0;
    }
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static int visit(Search *search) {
    ++search->nodes;
    if ((search->nodes & 1023) == 0 && monotonic_seconds() >= search->deadline) {
        search->stopped = 1;
    }
    return !search->stopped;
}

static int count_bits(Bits bits) {
    return __builtin_popcountll(bits);
}

static Bits landings(Bits occupied) {
    return (occupied + BOTTOM) & BOARD;
}

/* Locate gaps that complete a four. The sentinel at the top of each column
   prevents shifts from joining cells across the edge of the board. */
static Bits gaps_in_direction(Bits stones, int step) {
    Bits pair = stones & (stones << step);
    Bits left = (pair << step) & ((stones << (3 * step)) | (stones >> step));
    Bits right = (pair >> (2 * step)) & ((stones >> (3 * step)) | (stones << step));
    return left | right;
}

static Bits winning_cells(Bits stones, Bits occupied) {
    Bits vertical = (stones << 1) & (stones << 2) & (stones << 3);
    return (vertical | gaps_in_direction(stones, STRIDE - 1)
                     | gaps_in_direction(stones, STRIDE)
                     | gaps_in_direction(stones, STRIDE + 1)) & (BOARD ^ occupied);
}

static int aligned(Bits stones, int step) {
    Bits pairs = stones & (stones >> step);
    return (pairs & (pairs >> (2 * step))) != 0;
}

static int has_four(Bits stones) {
    return aligned(stones, STRIDE) || aligned(stones, 1)
        || aligned(stones, STRIDE - 1) || aligned(stones, STRIDE + 1);
}

static Bits nonlosing_moves(Bits legal, Bits enemy_wins) {
    Bits forced = legal & enemy_wins;
    if (forced) {
        if (forced & (forced - 1)) {
            return 0;
        }
        legal = forced;
    }
    return legal & ~(enemy_wins >> 1);
}

static int first_column(Bits moves) {
    for (int i = 0; i < WIDTH; ++i) {
        int column = center_order[i];
        if (moves & columns[column]) {
            return column;
        }
    }
    return -1;
}

static int order_moves(Move ordered[WIDTH], Bits moves, Bits current,
                       Bits occupied, int preferred) {
    int length = 0;
    for (int i = 0; i < WIDTH; ++i) {
        int column = center_order[i];
        Bits move = moves & columns[column];
        if (!move) {
            continue;
        }
        Bits threats = winning_cells(current | move, occupied | move);
        int priority = 8 * count_bits(threats) + 8 * count_bits(threats & landings(occupied | move));
        if (column == preferred) {
            priority += 1024;
        }
        int index = length++;
        while (index > 0 && priority > ordered[index - 1].priority) {
            ordered[index] = ordered[index - 1];
            --index;
        }
        ordered[index] = (Move){move, column, priority};
    }
    return length;
}

static Bits reflect_key(Bits key) {
    return ((key & (LOW_BITS(STRIDE) << 0)) << 42)
         | ((key & (LOW_BITS(STRIDE) << 7)) << 28)
         | ((key & (LOW_BITS(STRIDE) << 14)) << 14)
         |  (key & (LOW_BITS(STRIDE) << 21))
         | ((key & (LOW_BITS(STRIDE) << 28)) >> 14)
         | ((key & (LOW_BITS(STRIDE) << 35)) >> 28)
         | ((key & (LOW_BITS(STRIDE) << 42)) >> 42);
}

static Bits canonical_key(Bits current, Bits occupied, int played, int *mirrored) {
    /* In each column the sum identifies both its height and its ownership. */
    Bits key = current + occupied;
    *mirrored = 0;
    if (played < 12) {
        Bits reflected = reflect_key(key);
        if (reflected < key) {
            *mirrored = 1;
            return reflected;
        }
    }
    return key;
}

static unsigned hash_index(Bits key, int bits) {
    key ^= key >> 23;
    return (unsigned)((key * UINT64_C(0x9e3779b97f4a7c15)) >> (64 - bits));
}

static int proof_shift(int need_win) {
    return KEY_BITS + 4 * need_win;
}

static int proof_result(Bits entry, int need_win, int mirrored) {
    int result = (int)((entry >> proof_shift(need_win)) & 15);
    if (result >= SUCCESS && mirrored) {
        return SUCCESS + (WIDTH - 1) - (result - SUCCESS);
    }
    return result;
}

static Bits proof_store(Bits entry, int need_win, int result, int mirrored) {
    if (result >= SUCCESS && mirrored) {
        result = SUCCESS + (WIDTH - 1) - (result - SUCCESS);
    }
    int shift = proof_shift(need_win);
    return (entry & ~(UINT64_C(15) << shift)) | ((Bits)result << shift);
}

/* Test whether the side to move can force a win (need_win=1) or a draw or
   better (need_win=0). A safe move leaves the child without an immediate win. */
static int prove(Search *search, Bits current, Bits occupied, int played,
                 int need_win, int preferred) {
    if (!visit(search)) {
        return UNTESTED;
    }

    Bits other = current ^ occupied;
    Bits legal = landings(occupied);
    Bits safe = nonlosing_moves(legal, winning_cells(other, occupied));
    if (!safe) {
        return IMPOSSIBLE;
    }
    if (played >= CELLS - 2) {
        return need_win ? IMPOSSIBLE : SUCCESS + first_column(safe);
    }

    int mirrored;
    Bits key = canonical_key(current, occupied, played, &mirrored);
    unsigned slot = hash_index(key, CX_PROOF_BITS);
    Bits entry = proof_table[slot];
    if ((entry & KEY_MASK) != key) {
        entry = key;
    }
    int result = proof_result(entry, need_win, mirrored);
    if (result != UNTESTED) {
        return result;
    }

    if ((played & 1) == 0) {
        /* The responder can pair the initial squares of odd-height columns,
           then reply in the same column. The mover gets even-indexed rows
           plus at most one initial square in each pair. */
        Bits free = BOARD ^ occupied;
        Bits contested = legal & ODD_ROWS;
        Bits mover_may_get = (free & EVEN_ROWS) | contested;
        Bits responder_gets = free & ODD_ROWS & ~contested;
        if (!has_four(current | mover_may_get) && (need_win || has_four(other | responder_gets))) {
            result = IMPOSSIBLE;
        }
    }
    if (result == UNTESTED && played >= 26) {
        if (need_win && !has_four(BOARD ^ other)) {
            result = IMPOSSIBLE;
        } else if (!need_win && !has_four(BOARD ^ current)) {
            result = SUCCESS + first_column(safe);
        }
    }
    if (result != UNTESTED) {
        proof_table[slot] = proof_store(entry, need_win, result, mirrored);
        return result;
    }

    int other_result = proof_result(entry, 1 - need_win, mirrored);
    if (preferred < 0 && other_result >= SUCCESS) {
        preferred = other_result - SUCCESS;
    }

    Move ordered[WIDTH];
    int length;
    if ((safe & (safe - 1)) == 0) {
        ordered[0] = (Move){safe, first_column(safe), 0};
        length = 1;
    } else {
        length = order_moves(ordered, safe, current, occupied, preferred);
    }

    result = IMPOSSIBLE;
    for (int i = 0; i < length; ++i) {
        Move move = ordered[i];
        int reply = prove(search, other, occupied | move.move, played + 1, 1 - need_win, -1);
        if (search->stopped) {
            return UNTESTED;
        }
        if (reply == IMPOSSIBLE) {
            result = SUCCESS + move.column;
            break;
        }
    }

    entry = proof_store(entry, need_win, result, mirrored);
    if ((need_win && result >= SUCCESS) || (!need_win && result == IMPOSSIBLE)) {
        entry = proof_store(entry, 1 - need_win, result, mirrored);
    }
    proof_table[slot] = entry;
    return result;
}

static int solve(Search *search, Bits current, Bits occupied, int played,
                 int preferred, int *column) {
    Bits legal = landings(occupied);
    Bits wins = legal & winning_cells(current, occupied);
    *column = first_column(wins);
    if (wins || !legal) {
        return wins ? WIN : DRAW;
    }
    int result = prove(search, current, occupied, played, 1, preferred);
    if (result >= SUCCESS) {
        *column = result - SUCCESS;
        return WIN;
    }
    if (result == UNTESTED || monotonic_seconds() >= search->deadline) {
        return UNKNOWN;
    }
    result = prove(search, current, occupied, played, 0, preferred);
    if (result >= SUCCESS) {
        *column = result - SUCCESS;
        return DRAW;
    }
    return result == IMPOSSIBLE ? LOSS : UNKNOWN;
}

static int threat_value(Bits wins, Bits legal, Bits parity) {
    return 26 * count_bits(wins)
         + 29 * count_bits(wins & parity)
         + 32 * count_bits(wins & legal)
         + 13 * count_bits(wins & (legal << 1))
         + 20 * count_bits(wins & (wins << 1));
}

static int center_value(Bits stones) {
    return 7 * count_bits(stones & COLUMN(3))
         + 4 * count_bits(stones & (COLUMN(2) | COLUMN(4)))
         + 2 * count_bits(stones & (COLUMN(1) | COLUMN(5)));
}

/* Count unblocked lines with at least two stones in different positions. */
static int open_pairs(Bits own, Bits other, int step) {
    Bits free = BOARD ^ other;
    Bits open = free & (free >> step) & (free >> (2 * step)) & (free >> (3 * step));
    Bits second = own >> step;
    Bits third = own >> (2 * step);
    Bits fourth = own >> (3 * step);
    Bits pairs = (own & second) | (third & fourth) | ((own | second) & (third | fourth));
    return count_bits(open & pairs);
}

static int evaluate(Bits current, Bits other, Bits occupied, Bits legal,
                    Bits enemy_wins, int played) {
    Bits my_wins = winning_cells(current, occupied);
    Bits my_parity = BOTTOM * ((played & 1) ? 42 : 21);
    int score = threat_value(my_wins, legal, my_parity)
              - threat_value(enemy_wins, legal, BOARD ^ my_parity)
              + center_value(current) - center_value(other);

    for (int step = STRIDE - 1; step <= STRIDE + 1; ++step) {
        score += 4 * (open_pairs(current, other, step) - open_pairs(other, current, step));
    }
    score += 2 * (open_pairs(current, other, 1) - open_pairs(other, current, 1));
    if (score > HEURISTIC_LIMIT) {
        return HEURISTIC_LIMIT;
    }
    if (score < -HEURISTIC_LIMIT) {
        return -HEURISTIC_LIMIT;
    }
    return score;
}

static int negamax(Search *search, Bits current, Bits occupied, int played,
                   int depth, int alpha, int beta, int *principal) {
    if (!visit(search)) {
        return 0;
    }

    Bits other = current ^ occupied;
    Bits legal = landings(occupied);
    Bits enemy_wins = winning_cells(other, occupied);
    Bits safe = nonlosing_moves(legal, enemy_wins);
    if (!safe || played >= CELLS - 2) {
        if (principal) {
            *principal = first_column(safe ? safe : legal);
        }
        return safe || !legal ? 0 : -MATE + played + 2;
    }
    if (depth == 0 && (safe & (safe - 1))) {
        return evaluate(current, other, occupied, legal, enemy_wins, played);
    }

    int max_score = MATE - played - 3;
    if (beta > max_score) {
        beta = max_score;
        if (alpha >= beta) {
            return beta;
        }
    }

    int mirrored;
    Bits key = canonical_key(current, occupied, played, &mirrored);
    SearchEntry *slot = &search_table[hash_index(key, CX_SEARCH_BITS)];
    SearchEntry entry = *slot;
    int preferred = -1;
    if (entry.key == key && entry.bound) {
        preferred = mirrored ? WIDTH - 1 - entry.column : entry.column;
        if (entry.depth >= depth && (entry.bound == EXACT
                || (entry.bound == LOWER && entry.score >= beta)
                || (entry.bound == UPPER && entry.score <= alpha))) {
            if (principal) {
                *principal = preferred;
            }
            return entry.score;
        }
    }

    Move ordered[WIDTH];
    int length;
    if ((safe & (safe - 1)) == 0) {
        ordered[0] = (Move){safe, first_column(safe), 0};
        length = 1;
    } else {
        length = order_moves(ordered, safe, current, occupied, preferred);
    }

    int best = -INF;
    int best_column = ordered[0].column;
    int original_alpha = alpha;
    int next_depth = depth > 0 ? depth - 1 : 0;
    for (int i = 0; i < length; ++i) {
        Move move = ordered[i];
        int score;
        if (i > 0 && depth >= 4 && alpha + 1 < beta) {
            score = -negamax(search, other, occupied | move.move, played + 1,
                             next_depth, -alpha - 1, -alpha, 0);
            if (!search->stopped && score > alpha && score < beta) {
                score = -negamax(search, other, occupied | move.move, played + 1,
                                 next_depth, -beta, -alpha, 0);
            }
        } else {
            score = -negamax(search, other, occupied | move.move, played + 1,
                             next_depth, -beta, -alpha, 0);
        }
        if (search->stopped) {
            return 0;
        }
        if (score > best) {
            best = score;
            best_column = move.column;
        }
        if (best >= beta) {
            break;
        }
        if (best > alpha) {
            alpha = best;
        }
    }

    int bound = EXACT;
    if (best <= original_alpha) {
        bound = UPPER;
    } else if (best >= beta && best < max_score) {
        bound = LOWER;
    }
    if (slot->key != key || slot->depth <= depth) {
        int stored_column = mirrored ? WIDTH - 1 - best_column : best_column;
        *slot = (SearchEntry){key, (int16_t)best, (uint8_t)depth, (int8_t)stored_column, (uint8_t)bound};
    }
    if (principal) {
        *principal = best_column;
    }
    return best;
}

EXPORT int cx_prove(Bits current, Bits occupied, double seconds, int preferred,
                    int *column, uint64_t *nodes) {
    double start = monotonic_seconds();
    if (start == 0.0 || seconds <= 0.0) {
        return UNKNOWN;
    }
    Search search = {start + seconds, 0, 0};
    int outcome = solve(&search, current, occupied, count_bits(occupied), preferred, column);
    if (nodes) {
        *nodes = search.nodes;
    }
    return outcome;
}

EXPORT int cx_can_force(Bits current, Bits occupied, double seconds, int need_win, int preferred,
                        int *column) {
    double start = monotonic_seconds();
    if (start == 0.0 || seconds <= 0.0) {
        return UNKNOWN;
    }
    Bits legal = landings(occupied);
    Bits wins = legal & winning_cells(current, occupied);
    *column = first_column(wins);
    if (wins || !legal) {
        return wins || !need_win;
    }
    Search search = {start + seconds, 0, 0};
    int result = prove(&search, current, occupied, count_bits(occupied), need_win, preferred);
    if (result >= SUCCESS) {
        *column = result - SUCCESS;
        return 1;
    }
    return result == IMPOSSIBLE ? 0 : UNKNOWN;
}

EXPORT int cx_choose(Bits current, Bits occupied, double seconds, int *depth_reached,
                     uint64_t *nodes) {
    double start = monotonic_seconds();
    Bits legal = landings(occupied);
    Bits wins = legal & winning_cells(current, occupied);
    if (wins || !legal) {
        return first_column(wins);
    }
    Bits safe = nonlosing_moves(legal, winning_cells(current ^ occupied, occupied));
    int best = first_column(safe ? safe : legal);
    if ((safe & (safe - 1)) == 0 || seconds <= 0 || start == 0.0) {
        return best;
    }

    int played = count_bits(occupied);
    int remaining = CELLS - played;
    Search search = {start + seconds, 0, 0};
    if (remaining <= 32) {
        double fraction = 0.15;
        if (remaining <= 23) {
            fraction = 0.72;
        } else if (remaining <= 28) {
            fraction = 0.50;
        }
        Search proof = {start + seconds * fraction, 0, 0};
        int column;
        int outcome = solve(&proof, current, occupied, played, -1, &column);
        search.nodes = proof.nodes;
        if (outcome == WIN || outcome == DRAW) {
            if (depth_reached) {
                *depth_reached = remaining;
            }
            if (nodes) {
                *nodes = search.nodes;
            }
            return column;
        }
    }

    for (int depth = 2; depth <= remaining && monotonic_seconds() < search.deadline; ++depth) {
        int column = best;
        int score = negamax(&search, current, occupied, played, depth, -INF, INF, &column);
        if (search.stopped) {
            break;
        }
        best = column;
        if (depth_reached) {
            *depth_reached = depth;
        }
        if (score > HEURISTIC_LIMIT || (score < -HEURISTIC_LIMIT && depth >= 16)) {
            break;
        }
    }
    if (nodes) {
        *nodes = search.nodes;
    }
    return best;
}
