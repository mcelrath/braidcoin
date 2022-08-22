I'm writing this post to help organize and clarify some of the thought
surrounding "decentralized mining pools" for bitcoin, their requirements, and
some unsolved problems that must be solved before such a thing can be fully
specified.

# Decentralized mining pools for bitcoin

A decentralized mining pool consists of the following components:
1. A [weak block](#weak-blocks) and difficulty target mechanism
2. A [consensus mechanism](#consensus-mechanism) for collecting and accounting
   for shares
3. A [payout commitment](#payout-commitment) requiring a quorum of participants
   to sign off on share payments
4. A [signing procedure](#signing-procedure) for signing the [payout
   commitment](#payout-commitment) in such a way that the pool participants are
   properly paid.

In addition to this there are improvements being made by the
[StratumV2](https://github.com/stratum-mining/sv2-spec) project which include
transaction selection and encrypted communication to mining devices. These
problems are important but factorizable from the pool itself. StratumV2 is
solving the problem of decentralizing transaction selection while a
decentralized mining pool would already have decentralized transaction selection
and additionally solve the problem of decentralized and trustless payment for
shares.

# Weak Blocks

A *share* is a "weak block" that is defined as a standard bitcoin block that
does not meet bitcoin's target difficulty $T$, but does meet some lesser
difficulty target $t$. The *pool* specifies this parameter $t$ and when "weak
block" is found, it is communicated to the pool.

The share is itself a bearer proof that a certain amount of sha256 computation
has been done. The share itself must have a structure that indicates that it
"belongs" to a particular pool. In the case of centralized pools, this happens
because the pool itself hands out "work units" (bitcoin headers) corresponding
to a block that has been created by the pool, with transaction selection done
by the (centralized) pool.

In the case of a decentralized pool, the share itself must have additional
structure that indicates to other miners in the pool that the share belongs to
the pool, and if it had met bitcoin's difficulty target, the share contains
commitments such that all *other* miners in the pool would be paid according to
the share tally achieved by the decentralized [consensus
mechanism](#consensus-mechanism) of the pool.

Shares or blocks which do not commit to the additional metadata proving that the
share is part of the pool must be excluded from the share calculation, and those
miners are not "part of" the pool. In other words, submitting a random sha256
header to a pool must not count as a share contribution unless the ultimate
payout for that share, had it become a bitcoin block, would have paid the pool
in such a way that all other hashers are paid.

For example, consider a decentralized mining pool's "share" that looks like:

    Version | Previous Block Hash | Merkle Root | Timestamp | Difficulty Target | Nonce
    Coinbase Transaction | Merkle Sibling | Merkle Sibling | ...
    Pool Metadata

Here the `Merkle Siblings` in the second line are the additional nodes in the
transaction Merkle tree necessary to verify that the specified `Coinbase
Transaction` transaction is included in the `Merkle Root`. We assume that this
`Coinbase Transaction` commits to any additional data needed for the pool's
[consensus mechansim](#consensus-mechanism), for instance in an `OP_RETURN`
output.

The `Coinbase Transaction` is a standard transaction having no inputs, and
should have the following outputs:

    OutPoint(Value:0, scriptPubKey OP_RETURN <Pool Commitment>)
    OutPoint(Value:<block reward>, scriptPubKey <P2TR pool_pubkey>)

The `<block reward>` is the sum of all fees and block reward for this halving
epoch, and `pool_pubkey` is an address controlled collaboratively by the pool in
such a way that the [consensus mechanism](#consensus-mechanism) can only spend
it in such a way as to pay all hashers in the manner described by its share
accounting.

The `<Pool Commitment>` is a hash of `Pool Metadata` committing to any
additional data required for the operation of the pool. At a minimum, this must
include the weak difficulty target $t$ (or it must be computable from this
metadata). Validation of this share requires that the PoW hash of this bitcoin
header be less than this weak difficulty target $t$.

Other things that one might want to include in the `<Pool Commitment>` are:

1. Pubkeys of the hasher that can be used in collaboratively signing the
    [payout commitment](#payout-commitment),
2. Keys necessary for encrypted communication with this miner,
3. Identifying information such as an IP address, TOR address, or other routing
   information that would allow other miners to communicate out-of-band with
   this miner
4. Parents of this share (bead, in the case of braidpool), or other
   consensus-specific data if some other [consensus
   mechansim](#consensus-mechanism) is used.
5. Intermediate consensus data involved in multi-round threshold signing
   ceremonies.

Finally we note that there exists a proposal for [mining coordination using
covenants (CTV)](https://utxos.org/uses/miningpools/) that does not use weak
blocks, does not sample hashrate any faster than bitcoin blocks, and is
incapable of reducing variance. It is therefore not a "pool" in the usual sense
and we will not consider that design further, though covenants may still be
useful for a decentralized mining pool, which we discuss in [Payout
Commitments](#payout-commitments).

# Consensus Mechanism

In a centralized pool, the central pool operator receives all shares and does
accounting on them. While this accounting is simple, the point of a
decentralized mining pool is that we don't want to trust any single entity to do
this correctly, nor do we want to give any single entity control to steal all
funds in the pool, or the power to issue payments incorrectly.

Because of this, all hashers must receive the [shares](#weak-blocks) of all
other hashers. Each share could have been a bitcoin block if it had met
bitcoin's difficulty target and must commit to the pool metadata as described
above.

With the set of shares for a given epoch, we must place a consensus mechanism on
the shares, so that all participants of the pool agree that these are valid
shares and deserve to be paid out according to the pool's payout mechanism.

The consensus mechanism must have the characteristic that it operates much
*faster* than bitcoin, so that it can collect as many valid shares as possible
between valid bitcoin blocks. The reason for this is that one of the primary
goals of a pool is *variance reduction*.
[P2Pool](https://en.bitcoin.it/wiki/P2Pool) achieved this by using a standard
blockchain having a block time of 30s, and the [Monero
p2pool](https://github.com/SChernykh/p2pool) achieves it using a block time of
10s.

Bitcoin has spawned a great amount of research into consensus algorithms which
might be considered here including the [GHOST
protocol](https://eprint.iacr.org/2013/881), [asynchronous
PBFT](https://eprint.iacr.org/2016/199), "sampling" algorithms such as
[Avalanche](https://arxiv.org/abs/1906.08936) and that used by
[DFinity](https://arxiv.org/abs/1704.02397), and
[DAG-based](https://eprint.iacr.org/2018/104) algorithms. (This is not an
exhaustive bibliography but just a representative sample of the options)

One characteristic that is common to all consensus algorithms is that consensus
cannot be arrived at faster than roughly the global network latency. Regardless
of which consensus algorithm is chosen, it is necessary for all participants to
see all data going into the current state, and be able to agree that this is the
correct current state. Surveying the networks developed with the above
algorithms, one finds that the fastest they can come to consensus is in around
one second. Therefore the exact "time-to-consensus" of different algorithms
varies by an O(1) constant, all of them are around 1 second. This is around 600
times faster than bitcoin blocks, and results in miners 600 times smaller being
able to contribute, and a 600x factor reduction in variance compared to solo
mining.

While a 600x decrease in variance is a worthy goal, this is not enough of an
improvement to allow a single modern mining device to reduce its variance enough
to be worthwhile. Therefore, a different solution must be found for miners
smaller than a certain hashrate.

From our perspective, the obvious choice for a consensus algorithm is a DAG
which re-uses bitcoin's proof of work in the same spirit as bitcoin itself --
that is, the chain tip is defined by the heaviest work-weighted tip, and
conflict resolution within the DAG uses work-weighting. Note that this is the
same as the "longest chain rule" which only works at constant difficulty, but we
assume the DAG does not have constant difficulty so combining difficulties must
be done correctly. The solution is to identify the heaviest weight linear path
from genesis to tip, where the difficulties are summed along the path.

Finally, we caution that in considering mechanisms to include even smaller
miners, one must not violate the "progress-free" characteristic of bitcoin. That
is, one should not sum work from a subset of smaller miners to arrive at a
higher-work block in the DAG.

# Payout Commitment

# Unsolved Problems


this is
 a
