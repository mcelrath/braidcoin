#!/usr/bin/env python3

# Sample code for doing computations with braids
#
# The code here emphasizes clarity over speed.  We have used the memoize()
# function to memoize functions that are called repeatedly with the same
# arguments.  Use of memoize is an indication that better algorithms exist.

import hashlib
import bitcoin  # uses python-bitcoinlib https://github.com/petertodd/python-bitcoinlib
from bitcoin.core import uint256_from_str as uint256
import graph_tool.all as gt
import graph_tool.draw as gtdraw
import numpy as np
import scipy
import scipy.stats
from numpy.random import choice, sample, randint
from copy import copy
from functools import reduce
import itertools
from math import sqrt
from time import time

NETWORK_SIZE = 1.0  # The round-trip time in seconds to traverse the network
TICKSIZE = 0.1      # One "tick" of the network in which beads will be propagated and mined
DIFFICULTY = 6      # The likelihood that a node with target = 1 will mine a bead in a given tick
MAX_HASH = 2**256-1 # Maximum value a 256 bit hash can have, used to calculate difficulty

bead_color          = ( 27/255, 158/255, 119/255, 1)    # Greenish
genesis_color       = (217/255,  95/255,   2/255, 1)    # Orangeish
cohort_color        = (117/255, 112/255, 179/255, 1)    # Purplish
tip_color           = (231/255,  41/255, 138/255, 1)    # Pinkish
sibling_color       = (102/255, 166/255,  30/255, 1)    # Light Greenish
highlight1_color    = (      1,       1,       0, 1)    # Yellow
highlight2_color    = (      1,       0,       1, 1)    # Magenta
highlight3_color    = (      0,       1,       1, 1)    # Yellow
nohighlight_color   = (      1,       1,       1, 1)    # White
me_color            = (      0,       0,       0, 1)    # Black

descendant_color    = highlight2_color
ancestor_color      = highlight3_color

# A rotating color palette to color cohorts
color_palette = [genesis_color, cohort_color, sibling_color, tip_color]

#gencache = {}
#gencache[True] = {}
#gencache[False] = {}
cohort_size_benchmark = [] # cohort size vs time

def sha256(x: int): return hashlib.sha256(('%d'%x).encode()).digest()

def printvset(vs):
    """ Print a (sub-)set of vertices in compact form. """
    return("{"+",".join(sorted([str(v) for v in vs]))+"}")

class Network:
    """ Abstraction for an entire network containing <n> nodes.  The network has
        a internal clock for simulation which uses <ticksize>.  Latencies are taken
        from a uniform distribution on [0,1) so <ticksize> should be < 1.
    """
    def __init__(self, nnodes, difficulty=DIFFICULTY, ticksize=TICKSIZE, npeers=4, target=None):
        self.t = 0       # the current "time"
        self.ticksize = ticksize # the size of a "tick": self.t += self.tick at each step
        self.npeers = npeers
        self.nnodes = nnodes
        self.genesis = uint256(sha256(0))
        self.beads = {}  # a hash map of all beads in existence
        #self.inflightdelay = {} # list of in-flight beads
        #self.mempool = set() # A list of transactions for everyone to mine.  Everyone 
                             # sees the same mempool, p2p propegation delays are not modelled
        self.beads[self.genesis] = Bead(self.genesis, set(), set(), self, -1)
# FIXME not modelling mempool propegation means that we will either have all blocks in a round have
# the same tx, or none.  Maybe each round mining should grab a random subset?
        if not target:
            self.nodes = [Node(self.genesis, self, nodeid, 1/TICKSIZE, difficulty=difficulty) for nodeid in range(nnodes)]
        else:
            self.nodes = [Node(self.genesis, self, nodeid, 1/TICKSIZE, target=target) for nodeid in range(nnodes)]
        for (node, peers) in zip(self.nodes, [choice(list(set(range(nnodes)) - {me}),  \
                npeers, replace=False) for me in range(nnodes)]):
            #print("Node ", node, " has peers ", peers)
            node.setpeers([self.nodes[i] for i in peers])
        self.reset(difficulty=difficulty, target=target)

    def tick(self, mine=False):
        """ Execute one tick. """
        self.t += self.ticksize

        # Create a new set of transaction IDs in the mempool
        self.mempool.update([uint256(sha256(randint(2**63-1))) for dummy in range(randint(1,10))])

        # Have each node attempt to mine a random subset of the global mempool
        for node in self.nodes:
            # numpy.random.choice doesn't work with sets :-(
            node.tick(choice(list(self.mempool), randint(len(self.mempool)), replace=False), mine)

        for (node, bead) in copy(self.inflightdelay):
            self.inflightdelay[(node, bead)] -= self.ticksize
            if self.inflightdelay[(node, bead)] < 0: 
                node.receive(bead)
                del self.inflightdelay[(node, bead)]

    def broadcast(self, node, bead, delay):
        """ Announce a block/bead discovery to a node who is <delay> away. """
        if bead not in node.beads:
            prevdelay = NETWORK_SIZE
            if (node,bead) in self.inflightdelay: prevdelay = self.inflightdelay[(node, bead)]
            self.inflightdelay[(node, bead)] = min(prevdelay, delay)

    def reset(self, difficulty=DIFFICULTY, target=None):
        self.beads = {}
        self.beads[self.genesis] = Bead(self.genesis, set(), set(), self, -1)
        self.inflightdelay = {}
        self.mempool = set()
        for node in self.nodes:
            node.reset(difficulty, target)

    def printinflightdelays(self):
        for (node, bead) in self.inflightdelay:
            print("bead ", bead, " to node ", node, " will arrive in %fs"%self.inflightdelay[(node, bead)])

class Node:
    """ Abstraction for a node. """
    def __init__(self, genesis, network, nodeid, hashrate, difficulty=None, target=None):
        self.genesis = genesis
        self.network = network
        self.peers = []
        self.latencies = []
        self.nodeid = nodeid
        self.nodesalt = uint256(sha256(randint(2**63-1)))
        self.nonce = 0      # Will be increased in the mining process
        self.hashrate = hashrate
        self.reset(difficulty, target)

    def reset(self, difficulty=None, target=None):
        self.beads = [self.network.beads[self.network.genesis]]     # A list of beads in the order received
        self.braids = [Braid(self.beads)]  # A list of viable braids, each having a braid tip
        self.mempool = set()  # A set of txids I'm mining
        self.incoming = set() # incoming beads we were unable to process
        if difficulty:
            self.target = 1 << (256 - difficulty - randint(4))
        elif target:
            self.target = target
        else:
            self.target = 1 << (256-5)
        #print(self.braids[0].beads)
        self.braids[0].tips = {self.beads[0]}
        self.hremaining = np.random.geometric(self.target/MAX_HASH) 

    def __str__(self):
        return "<Node %d>"%self.nodeid

    def setpeers(self, peers, latencies=None):
        """ Add a peer separated by a latency <delay>. """
        self.peers = peers
        if latencies: self.latencies = latencies
        else:         self.latencies = sample(len(peers))*NETWORK_SIZE
        assert(len(self.peers) == len(self.latencies))

    def tick(self, newtxs=[], mine=True):
        """ Add a Bead satisfying <target>. """
        # First try to extend all braids by received beads that have not been added to a braid
        newincoming = set()
        oldtips = self.braids[0].tips
        while len(newincoming) != len(self.incoming):
            for bead in self.incoming:
                for braid in self.braids:
                    if not braid.extend(bead):
                        newincoming.add(bead)
            self.incoming = newincoming
        if mine:
            PoW = uint256(sha256(self.nodesalt**3+self.nonce))
            self.nonce += 1
            if PoW < self.target:
                b = Bead(PoW, copy(self.braids[0].tips), copy(self.mempool), self.network, self.nodeid)
                self.receive(b)     # Send it to myself (will rebroadcast to peers)
                # TODO remove txids from mempool
        else :
            self.hremaining -= self.hashrate*self.network.ticksize
            if(self.hremaining <= 0):
                PoW = (uint256(sha256(self.nodesalt**3+self.nonce))*self.target)//MAX_HASH
                self.nonce += 1
                # The expectation of how long it will take to mine a block is Geometric
                # This node will generate one after this many hashing rounds (ticks)
                b = Bead(PoW, copy(self.braids[0].tips), copy(self.mempool), self.network, self.nodeid)
                self.receive(b)     # Send it to myself (will rebroadcast to peers)
                self.hremaining = np.random.geometric(self.target/MAX_HASH) 
            elif(self.braids[0].tips != oldtips):
                # reset mining if we have new tips
                self.hremaining = np.random.geometric(self.target/MAX_HASH) 

    def receive(self, bead):
        """ Recieve announcement of a new bead. """
        # TODO Remove txids from mempool
        #print("node ", self, " receive()ing ", bead)
        if bead in self.beads: return
        else: self.beads.append(bead)
        #print("Node ", self, " is receiving ", bead)
        for braid in self.braids:
            if not braid.extend(bead):
                self.incoming.add(bead) # index of vertex is index in list
        #print("Rebroadcasting ", bead, " to peers ", [self.peers[p].nodeid for p in range(self.network.npeers)])
        self.send(bead)

    def send(self, bead):
        """ Announce a new block from a peer to this node. """
        #print("send()ing bead ", bead, " to peers ", [self.peers[p].nodeid for p in range(self.network.npeers)])
        for (peer, delay) in zip(self.peers, self.latencies):
            self.network.broadcast(peer, bead, delay)

class Bead:
    """ A bead is either a block of transactions, or an individual transaction.
        This class stores auxiliary information about a bead and is separate
        from the vertex being stored by the Braid class.  Beads are stored by
        the Braid object in the same order as vertices.  So if you know the
        vertex v, the Bead instance is Braid.beads[int(v)].  graph_tool vertices
        can be cast to integers as int(v), giving their index.
    """

    # FIXME lots of stuff here
    def __init__(self, hash, parents, transactions, network, creator):
        self.t = network.t
        self.hash = hash    # a hash that identifies this block
        self.parents = parents
        self.children = set() # filled in by Braid.make_children()
        self.siblings = set() # filled in by Braid.analyze
        self.cohort = set() # filled in by Braid.analyze
        self.transactions = transactions
        self.network = network
        self.creator = creator
        if creator != -1: # if we're not the genesis block (which has no creator node)
            self.difficulty = MAX_HASH/network.nodes[creator].target
        else: self.difficulty = 1
        self.sibling_difficulty = 0
        network.beads[hash] = self # add myself to global list
        self.reward = None  # this bead's reward (filled in by Braid.rewards)

    def __str__(self):
        return "<Bead ...%04d>"%(self.hash%10000)

class Braid(gt.Graph):
    """ A Braid is a Directed Acyclic Graph with no incest (parents may not also
        be non-parent ancestors).  A braid may have multiple tips. """

    def __init__(self, beads=[]):
        super().__init__(directed=True, vorder=True)
        self.times = self.new_vertex_property("double")
        self.beads = []                                           # A list of beads in this braid
        self.vhashes = {}                                         # A dict of (hash, Vertex) for each vertex
        self.vcolors = self.new_vertex_property("vector<float>")  # vertex colorings
        self.vhcolors = self.new_vertex_property("vector<float>") # vertex halo colorings
        self.vsizes = self.new_vertex_property("float")           # vertex size
        self.cohort_sizes = []
        self.cohort_times = []
        self.ncohorts = 0                                         # updated by cohorts()

        if beads:
            for b in beads: 
                self.beads.append(b)  # Reference to a list of beads
                self.vhashes[b.hash] = self.add_vertex()
                self.vhashes[b.hash].bead = b
            # FIXME add edges if beads has more than one element.
            self.tips = {beads[-1]}
        self.tips = set()      # A tip is a bead with no children.
        self.acache = {}
        self.dcache = {}

    def extend(self, bead):
        """ Add a bead to the end of this braid. Returns True if the bead
            successfully extended this braid, and False otherwise. """
        if (not bead.parents                                            # No parents -- bad block
                or not all([p.hash in self.vhashes for p in bead.parents]) # We don't have all parents
                or bead in self.beads):                                # We've already seen this bead
            return False
        self.beads.append(bead)
        self.vhashes[bead.hash]                 = self.add_vertex()
        self.vhashes[bead.hash].bead            = bead
        self.vcolors[self.vhashes[bead.hash]]   = bead_color
        self.vhcolors[self.vhashes[bead.hash]]  = nohighlight_color
        self.vsizes[self.vhashes[bead.hash]]    = 12
        for p in bead.parents:
            self.add_edge(self.vhashes[bead.hash], self.vhashes[p.hash])
            self.times[self.vhashes[bead.hash]] = bead.t
            if p in self.tips:
                self.tips.remove(p)
        self.tips.add(bead)
        return True

    def printvset(self, vs):
        """ Print a (sub-)set of vertices in compact form. """
        print("{"+",".join([str(v) for v in vs])+"}")

    def rewards(self, coinbase):
        """ Compute the rewards for each bead, where each cohort is awarded
            <conbase> coins.
            FIXME splitting of tx fees not implemented.
        """
        for cohort in self.cohorts():
            for c in cohort:
                siblings = cohort - self.ancestors(c, cohort) - self.descendants(c, cohort) - {c}
                bc = self.beads[int(c)]
                # Denominator (normalization) for distribution among siblings
                bc.sibling_difficulty = MAX_HASH/(sum([self.beads[int(s)].difficulty for s in siblings]) 
                    + bc.difficulty)
            N = sum([self.beads[int(c)].difficulty/self.beads[int(c)].sibling_difficulty for c in cohort])
            for c in cohort:
                bc = self.beads[int(c)]
                bc.reward = coinbase*(bc.difficulty/bc.sibling_difficulty)/N

# FIXME I can make 3-way siblings too: find the common ancestor of any 3 siblings
# and ask what its rank is...
    def siblings(self, cohort):
        """ The siblings of a bead are other beads for which it cannot be
            decided whether the come before or after this bead in time.  
            Note that it does not make sense to call siblings() on a cohort 
            which contains dangling chain tips.  The result is a dict of 
                (s,v): (m,n) 
            which can be read as: 
                The sibling $s$ of vertex $v$ has a common ancestor $m$
                generations away from $v$ and a common descendant $n$
                generations away from $v$.
            """
        retval = dict()
        # Since siblings are mutual, we could compute (s,c) and (c,s) at the same time
        for c in cohort: 
            siblings = cohort - self.ancestors(c, cohort) - self.descendants(c, cohort) - {c}
            for s in siblings:
                ycas = self.youngest_common_ancestors({s,c})
                ocds = self.oldest_common_descendants({s,c})
                # Step through each generation of parents/children until the common ancestor is found
                pgen = {s} # FIXME either c or s depending on whether we want to step from the 
                for m in range(1,len(cohort)):
                    pgen = {q for p in pgen for q in self.parents(p) }
                    if pgen.intersection(ycas) or not pgen: break
                cgen = {s} # FIXME and here
                for n in range(1,len(cohort)):
                    cgen = {q for p in cgen for q in self.children(p)}
                    if cgen.intersection(ocds) or not cgen: break
                retval[int(s),int(c)] = (m,n)
        return retval

    def cohorts(self, initial_cohort=None, older=False):
        # given the seed of the next cohort (which is the set of beads one step
        # older, in the next cohort), build an ancestor and descendant set for
        # each visited bead.
        cohort      = initial_cohort or frozenset([self.vertex(0)])
        ancestors   = {} # for each visited bead
        lastgen     = cohort
        head        = cohort
        self.ncohorts = 0
        self.cohort_cache = []
        # on each new visited bead, see if it is in the descendant set for all
        # beads in the head.
        while True :
            self.ncohorts += 1
            self.cohort_cache.append(cohort)
            yield cohort
            head        = self.next_generation(cohort, older) - cohort
            ancestors   = {h: frozenset() for h in head}
            parents     = ancestors
            lastgen     = head
            while True :
                thisgen = self.next_generation(lastgen, older)
                if not thisgen: return # Ends the iteration (StopIteration)
                for v in thisgen:
                    parents[v] = self.next_generation(v, not older)
                    if all([p in ancestors for p in parents[v]]):
                        ancestors[v] = parents[v].union(*[ancestors[p] for p in parents[v]])
                if(all([p in ancestors] for p in frozenset.union(*[parents[v] for v in thisgen]))# we have no missing ancestors
                    and all([h in ancestors[v] for h in head for v in thisgen])):                  # and everyone has all head beads as ancestors
                    cohort = frozenset.intersection(*[ancestors[v] for v in thisgen])
                    nexthead = self.next_generation(cohort, older) - cohort
                    tail = self.next_generation(nexthead, not older)    # the oldest beads in the candidate cohort
                    if all([p in ancestors[n] for n in nexthead for p in tail]):
                        break
                lastgen = thisgen

    def cohort_time(self):
        """ Compute the average cohort time and its standard deviation returned
            as a tuple (mean, stddev). """
        t = 0
        ctimes = []
        for c in self.cohorts():
            if c == {self.vertex(0)}: continue # skip genesis bead
            times = [self.beads[int(v)].t for v in c]
            ctimes.append(max(times)-t)
            t = max(times)
        return (np.mean(ctimes), np.std(ctimes))

    def exclude(self, vs, predicate):
        """ Recursively exclude beads which satisfy a predicate (usually either
            parents or children) -- this removes all ancestors or descendants. """
        lastvs = copy(vs)
        while True:
            newvs = {v for v in vs if predicate(v) not in vs}
            if newvs == lastvs: return newvs
            lastvs = newvs
    
    def common_generation(self, vs, older=True):
        """ Find the first common ancestor/descendant generation of all vs, and
        all intermediate ancestors/descendants by bfs.  This is analagous to the
        Most Recent Common Ancestor (MRCA) in biology.  The first return value
        should be the seed for the *next* cohort while the second return value
        is the *current* cohort.  """
        if older: (edgef, nodef, nextgen_f) = ("out_edges","target", self.parents)
        else:     (edgef, nodef, nextgen_f) = ("in_edges", "source", self.children)
        if not isinstance(vs, set): vs = {vs}
        lastvs = self.exclude(vs, nextgen_f)
        nextgen = {v: nextgen_f(v) for v in lastvs}
        lastgen = copy(nextgen)
        firstv = next(iter(lastvs))
        niter = 0
        while True:
            commond = set.intersection(*[nextgen[v] for v in nextgen]) - lastvs
            if commond: return commond
                #foo = self.exclude(commond, nextgen_f)
                #if foo != commond:
                #    print("self.exclude(commond, nextgen_f) removed something: ")
                #    self.printvset(foo)
                #    self.printvset(self.commond)
                #return foo
            else: # add one generation of descendants for bfs
                nextgenupd = dict()
                for v in lastgen:
                    nextgenupd[v] = nextgen_f(lastgen[v])
                    nextgen[v].update(nextgenupd[v])
                # We hit a tip, on all paths there can be no common descendants
                if not all([nextgenupd[v] for v in nextgenupd]): 
                    return set()
                lastgen = nextgen
            niter += 1
            if niter > 1000:
                raise Exception("infinite loop in common_generation? ")

    def oldest_common_descendants(self, vs):
        return self.common_generation(vs, older=False)

    def youngest_common_ancestors(self, vs):
        return self.common_generation(vs, older=True)

    def all_generations(self, v:gt.Vertex, older, cohort=None, limit=None):
        """ Return the next generation older or younger depending on the value
            of <older>. """
#        global gencache
#        if int(v) in gencache[older]: return gencache[older][v]
        if older: (edgef, nodef) = ("out_edges","target")
        else:     (edgef, nodef) = ("in_edges", "source")
        if cohort and v not in cohort: 
#            gencache[older][v] = set()
            return set()
        result = {x for x in [getattr(y, nodef)() for y in getattr(v, edgef)()]}
        if cohort: result = result.intersection(cohort)
        if not result: 
#            gencache[older][v] = set()
            return set()
        for x in copy(result): 
            if limit:
                result.update(self.all_generations(x, older, cohort, limit-1))
            else:
                result.update(self.all_generations(x, older, cohort, limit))
#        gencache[older][v] = result
        return result

    # FIXME this is very time consuming -- it could be greatly sped up by
    # recording ancestors when they are visited.
    def ancestors(self, v:gt.Vertex, cohort=None, limit=None):
        if v not in self.acache:
            self.acache[v] = self.all_generations(v, older=True, cohort=cohort, limit=limit)
        return self.acache[v]

    def descendants(self, v:gt.Vertex, cohort=None, limit=None):
        if v not in self.dcache: 
            self.dcache[v] = self.all_generations(v, older=True, cohort=cohort, limit=limit)
        return self.dcache[v]

    def next_generation(self, vs, older):
        if older: (edgef, nodef) = ("out_edges","target")
        else:     (edgef, nodef) = ("in_edges", "source")
        if isinstance(vs, gt.Vertex):
            return frozenset([getattr(y, nodef)() for y in getattr(vs,edgef)()])
        elif isinstance(vs, frozenset):
            ng = [self.next_generation(v, older) for v in vs]
            if not ng: return frozenset()
            else: return frozenset.union(*ng)

    def parents(self, vs):
        return self.next_generation(vs, older=True)

    def children(self, vs):
        return self.next_generation(vs, older=False)

    def plot(self, focusbead=None, cohorts=True, focuscohort=None, numbervertices=False,
            highlightancestors=False, output=None, rewards=False):
        """ Plot this braid, possibly coloring graph cuts.  <focusbead>
            indicates which bead to consider for coloring its siblings and
            cohort. """
        ancestors = set()
        descendants = set()
        vlabel = self.new_vertex_property("string")
        kwargs = {'vertex_size': self.vsizes, 
                  'vertex_font_size':8,
                  'nodesfirst':True,
                  #vertex_text':numbervertices and self.vertex_index or None, # FIXME use rank instead if focusbead
                  'vertex_text':vlabel,
                  'vertex_halo':True, 
                  'vertex_halo_size':0, 
                  'vertex_fill_color':self.vcolors,
                  'vertex_halo_color':self.vhcolors}
        if rewards:
            # We want the sum of the area of the beads to be a constant.  Since
            # the area is pi r^2, the vertex size should scale like the sqrt of
            # the reward
            self.rewards(400)
            for v in self.vertices():
                print(v)
                self.vsizes[v] = sqrt(self.beads[int(v)].reward)
        if output: kwargs['output'] = output
        if focusbead:
            ancestors   = self.ancestors(focusbead)
            descendants = self.descendants(focusbead)
            kwargs['vertex_halo_size'] = 1.5

            for v in self.vertices():
                # Decide the bead's color
                if v.out_degree() == 0:
                    self.vcolors[v] = genesis_color
                elif v.in_degree() == 0:
                    self.vcolors[v] = tip_color
                else:
                    self.vcolors[v] = bead_color

                # Decide the highlight color
                if v == focusbead:
                    self.vhcolors[v] = me_color
                elif v in ancestors and highlightancestors:
                    self.vhcolors[v] = ancestor_color
                elif v in descendants and highlightancestors:
                   self.vhcolors[v] = descendant_color
                else:
                    self.vhcolors[v] = nohighlight_color

            # Label our siblings with their rank
            for cohort in self.cohorts(): 
                if focusbead in cohort:
                    for c in cohort:
                        self.vcolors[c] = cohort_color
                    for (s,v),(m,n) in self.siblings(cohort).items():
                        if v == focusbead:
                            vlabel[self.vertex(s)] = "%d,%d"%(m,n)
                            #self.vcolors[s] = sibling_color
                    break
        else:
            cnum = 0
            #ncohorts = len(list(self.cohorts()))
            if cohorts:
                for c in self.cohorts():
                    for v in c:
                        if focuscohort == cnum:
                            self.vhcolors[v] = highlight1_color
                        self.vcolors[v] = color_palette[cnum%len(color_palette)]
                        self.vhcolors[v] = nohighlight_color
                    cnum += 1
            if numbervertices: kwargs['vertex_text'] = self.vertex_index

        return gtdraw.graph_draw(self, **kwargs)

