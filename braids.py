#!/usr/bin/env python3

import hashlib
import bitcoin
from bitcoin.core import uint256_from_str as uint256
import graph_tool.all as gt
import graph_tool.draw as gtdraw
import numpy as np
from numpy.random import choice, sample, randint
from copy import copy
from functools import reduce
import itertools

NETWORK_SIZE = 1.0  # The round-trip time in seconds to traverse the network
TICKSIZE = 0.1      # One "tick" of the network in which beads will be propagated and mined
DIFFICULTY = 6     # The likelihood that a node with target = 1 will mine a bead in a given tick

bead_color          = ( 27/255, 158/255, 119/255, 1)    # Greenish
sibling_color       = (217/255,  95/255,   2/255, 1)    # Orangeish
cohort_color        = (117/255, 112/255, 179/255, 1)    # Purplish
tip_color           = (231/255,  41/255, 138/255, 1)    # Pinkish
genesis_color       = (102/255, 166/255,  30/255, 1)    # Light Greenish
highlight1_color    = (      1,       1,       0, 1)    # Yellow
highlight2_color    = (      1,       0,       1, 1)    # Magenta
nohighlight_color   = (      1,       1,       1, 1)    # White
me_color            = (      0,       0,       0, 1)    # Black

ancestor_color      = highlight1_color
descendant_color    = highlight2_color

# A rotating color palette to color cohorts
color_palette = [sibling_color, cohort_color, tip_color, genesis_color]

def sha256(x: int): return hashlib.sha256(('%d'%x).encode()).digest()

def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:            
            memo[x] = f(x)
        return memo[x]
    return helper

class Network:
    """ Abstraction for an entire network containing <n> nodes.  The network has
        a internal clock for simulation which uses <ticksize>.  Latencies are taken
        from a uniform distribution on [0,1) so <ticksize> should be < 1.
    """
    def __init__(self, nnodes, difficulty=DIFFICULTY, ticksize=TICKSIZE, npeers=4):
        self.t = 0       # the current "time"
        self.ticksize = ticksize # the size of a "tick": self.t += self.tick at each step
        self.npeers = npeers
        self.nnodes = nnodes
        self.beads = {}  # a hash map of all beads in existence
        self.genesis = uint256(sha256(0))
        self.mempool = set() # A list of transactions for everyone to mine.  Everyone 
                             # sees the same mempool, p2p propegation delays are not modelled
# FIXME not modelling mempool propegation means that we will either have all blocks in a round have
# the same tx, or none.  Maybe each round mining should grab a random subset?
        self.beads[self.genesis] = Bead(self.genesis, set(), set(), self, nnodes/2.0)
        self.nodes = [Node(self.genesis, self, difficulty, nodeid) for nodeid in range(nnodes)]
        self.inflightdelay = {} # list of in-flight beads
        for (node, peers) in zip(self.nodes, [choice(list(set(range(nnodes)) - {me}),  \
                npeers, replace=False) for me in range(nnodes)]):
            #print("Node ", node, " has peers ", peers)
            node.setpeers([self.nodes[i] for i in peers])

    def tick(self, mine=True):
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

    def printinflightdelays(self):
        for (node, bead) in self.inflightdelay:
            print("bead ", bead, " to node ", node, " will arrive in %fs"%self.inflightdelay[(node, bead)])

class Node:
    """ Abstraction for a node. """
    def __init__(self, genesis, network, difficulty, nodeid):
        self.genesis = genesis
        self.network = network
        self.peers = []
        self.latencies = []
        self.beads = [network.beads[network.genesis]]     # A list of beads in the order received
        self.braids = [Braid(self.beads)]  # A list of viable braids, each having a braid tip
        self.mempool = set()  # A set of txids I'm mining
        self.incoming = set() # incoming beads we were unable to process
        self.nodeid = nodeid
        self.nodesalt = uint256(sha256(randint(2**63-1)))
        self.nonce = 0      # Will be increased in the mining process
        self.target = 1 << (256 - difficulty - randint(4))

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
        from the vertex being stored by the Braid class. """

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
        network.beads[hash] = self # add myself to global list

    def __str__(self):
        return "<Bead ...%04d>"%(self.hash%10000)

class Braid(gt.Graph):
    """ A Braid is a Directed Acyclic Graph with no incest (parents may not also
        be non-parent ancestors).  A braid may have multiple tips. """

# FIXME it seems to me that a braid could be held as a set of beads whcih cannot
# be moved past one anther.  e.g. at vertex cuts.
    def __init__(self, beads=[]):
        super().__init__(directed=True, vorder=True)
        #super().set_directed(True) # Graphs are directed by default
        self.times = self.new_vertex_property("double")
        self.beads = []     # A list of beads in this braid
        self.vhashes = {}   # A dict of (hash, Vertex) for each vertex
        self.vcolors = self.new_vertex_property("vector<float>") # vertex colorings
        self.vhcolors = self.new_vertex_property("vector<float>") # vertex halo colorings

        if beads:
            for b in beads: 
                self.beads.append(b)  # Reference to a list of beads
                self.vhashes[b.hash] = self.add_vertex()
                self.vhashes[b.hash].bead = b
            # FIXME add edges if beads has more than one element.
        self.tips = set()      # A tip is a bead with no children.  FIXME this
                            # could also be an integer, showing how many tips at
                            # the end of self.beads are childless
        self.tips = {beads[-1]}

    #def vertex(bead: Bead):
    #    return self.vhashes[bead.hash]

    def extend(self, bead):
        """ Add a bead to the end of this braid. Returns True if the bead
            successfully extended this braid, and False otherwise. """
        if (not bead.parents                                            # No parents -- bad block
                or not all([p.hash in self.vhashes for p in bead.parents]) # We don't have all parents
                or bead in self.beads):                                # We've already seen this bead
            return False
        self.beads.append(bead)
        self.vhashes[bead.hash] = self.add_vertex()
        self.vhashes[bead.hash].bead = bead
        self.vcolors[self.vhashes[bead.hash]]     = bead_color
        self.vhcolors[self.vhashes[bead.hash]]    = nohighlight_color
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

    # FIXME remove?  not used
    def make_children(self):
        """ Walk the Braid and make children links for each Bead. """
        for bead in self.beads:
            for parent in bead.parents:
                parent.children.add(bead)

    def analyze(self):
        ancestors = set()
        descendants = set(self.vertices())
        all_vertices = set(self.vertices())
        for v in self.vertices():
            ocds = self.oldest_common_descendants(self.children(v))
            for c in [x.target() for x in v.out_edges()]:
                newancestors = copy(ancestors) + {c}
                newdescendants = copy(descendants) - self.children(v) # remove all siblings
                # Find the first common descendant of all children of v.
            cohort = all_vertices - descendants - ancestors
            print(cohort)

    def siblings(self, v):
        """ The siblings of a bead are defined as all the beads that are
            descendants of any parent, and ancestors of any grandchild of a
            parent.  """
        # FIXME return a dict where keys are vertices, and values are tuples of
        # the rank (m,n)
        parents = {e.source() for e in v.in_edges()}
        siblings = reduce(set.union, [{x.target() for x in {e for e in p.out_edges()}} for p in parents])
        grandchildren = reduce(set.union, [{x.target() for x in {e for e in s.out_edges()}} for s in siblings])
        lastsiblings = copy(siblings)
        while True:
            nextsiblings = reduce(set.union, [{x.target() for x in {e for e in s.out_edges()}} for s in lastsiblings])
            nextsiblings -= grandchildren
            if not nextsiblings: return siblings
            siblings |= nextsiblings
            lastsiblings = nextsiblings

    def cohorts(self, cohort=None, older=False):
        """ The cohort of a bead v are the siblings of v plus its sibling's
            siblings, recursively.  """
        if older: (edgef, nodef, nextgen_f) = ("out_edges","target", self.parents)
        else:     (edgef, nodef, nextgen_f) = ("in_edges", "source", self.children)
        ncohorts = 1
        vertices = cohort or {self.vertex(0)}
        while True:
            #print("================ next cohort: {"+",".join([str(v) for v in vertices])+"}")
            yield vertices
            # Find all children of the current cohort that are not in the cohort
            nextcohort = reduce(set.union, [{getattr(x, nodef)() for x in 
                {e for e in getattr(v,edgef)()} if getattr(x,nodef)() not in vertices} 
                for v in vertices])
            # Recursively expand cohort by adding siblings of existing beads
            lastgen = set()
            while True: 
                # We must compute the common generation with each iteration
                # because it's possible that the oldest common descendant is a
                # sibling with another child.
                cgen =  self.common_generation(nextcohort, older)
                nextgen = reduce(set.union, [{getattr(x,nodef)() for x in {e for e in
                    getattr(s,edgef)()} if getattr(x,nodef)() not in cgen} for s in nextcohort])
                if not nextgen or nextgen == lastgen: break
                nextcohort.update(nextgen)
                lastgen = nextgen
            vertices = nextcohort
            if not vertices: 
                self.ncohorts = ncohorts
                return
            ncohorts += 1

    # FIXME implement, and cache results
    @memoize
    def rank(self, u, v):
        """ Return a tuple (m,n) of the rank of the vertex u with respect to v.
            Note rank is not symmetric: rank(u,v) != rank(v,u) in general. """
        if u == v: return (0,0)

    def sibling_children(self, v):
        """ Return all descendants until we find a subgraph cut relative to v by bfs. """
        descendants = {e.target() for e in v.out_edges()}
        if len(descendants) == 1: return set()
        ocds = self.oldest_common_descendants(v)

        lastgen = copy(descendants)
        for c in lastgen:
            nextgen = {x.target() for x in c.out_edges()}
            nextgen -= ocds # Remove common descendants (termination)
            descendants |= nextgen
            lastgen = nextgen
        return descendants

    def exclude(self, vs, predicate):
        lastvs = copy(vs)
        while True:
            newvs = {v for v in vs if predicate(v) not in vs}
            if newvs == lastvs: return newvs
            lastvs = newvs

    def common_generation(self, vs, older=True):
        """ Find the first common ancestor/descendant generation of all vs by bfs. """
        if older: (edgef, nodef, nextgen_f) = ("out_edges","target", self.parents)
        else:     (edgef, nodef, nextgen_f) = ("in_edges", "source", self.children)
        if not isinstance(vs, set): vs = {vs}
        lastvs = self.exclude(vs, nextgen_f)
#print("ocds of {"+",".join([str(v) for v in lastvs])+"} are ", end="")
        nextgen = {v: nextgen_f(v) for v in lastvs}
        lastgen = copy(nextgen)
        firstv = next(iter(lastvs))
        while True:
            commond = set.intersection(*[nextgen[v] for v in nextgen]) - lastvs
            if commond: return self.exclude(commond, nextgen_f)
#print("{"+",".join([str(v) for v in self.exclude(commond, nextgen_f)])+"}")
            else: # add one generation of descendants for bfs
                nextgenupd = dict()
                for v in lastgen:
                    nextgenupd[v] = nextgen_f(lastgen[v])
                    nextgen[v].update(nextgenupd[v])
                # We hit a tip, on all paths there can be no common descendants
                if not any([nextgenupd[v] for v in nextgenupd]): return set() 
#print("{}")
                lastgen = nextgen

    def oldest_common_descendants(self, vs):
        return self.common_generation(vs, older=False)

    def youngest_common_ancestors(self, vs):
        return self.common_generation(vs, older=True)

#    # FIXME this can be generalized to find either the youngest common ancestor
#    # or oldest common descendant with a direction flag.
#    def oldest_common_descendants(self, vs):
#        """ Find the first common descendants of all vs by bfs. """
#        if not isinstance(vs, set): vs = {vs}
#        youngestvs = self.exclude(vs, self.children)
#        #print("ocds of {"+",".join([str(v) for v in youngestvs])+"} are ", end="")
#        descendants = {v: self.children(v) for v in youngestvs}
#        lastgen = copy(descendants)
#        firstv = next(iter(youngestvs))
#        while True:
#            commond = set.intersection(*[descendants[v] for v in descendants]) - youngestvs
#            if commond: return self.exclude(commond, self.children)
#            else: # add one generation of descendants for bfs
#                nextgen = dict()
#                for v in lastgen:
#                    nextgen[v] = self.children(lastgen[v])
#                    descendants[v].update(nextgen[v])
#                # We hit a tip, on all paths there can be no common descendants
#                if not any([nextgen[v] for v in nextgen]): return set() 
#                lastgen = nextgen
#
#    def youngest_common_ancestors(self, vs):
#        """ Find the first common ancestors of all vs by bfs. """
#        # FIXME fails if elements of vs are ancestors of each other
#        # Genesis bead special case
#        if not isinstance(vs, set): vs = {vs}
#        ancestors = dict()
#        for v in vs:
#            ancestors[v] = self.parents(v)
#            if not ancestors[v]: return set() # The genesis bead is in vs
#        #print(ancestors)
#        if not ancestors: return ancestors # No ancestors, empty set
#        lastgen = copy(ancestors)
#        firstv = next(iter(vs))
#        while True:
#            commona = set.intersection(*[ancestors[v] for v in ancestors]) - vs
#            if commona: return commona
#            else:
#                # add one generation of ancestors for bfs
#                nextgen = {}
#                for v in lastgen:
#                    nextgen[v] = self.parents(lastgen[v])
#                    ancestors[v].update(nextgen[v])
#                lastgen = nextgen

    def old_analyze(self, startv=0, highlight=0):
        """ Perform analysis on the Braid, filling in various data structures. """
        #for bead in self.beads:
        #    for parent in bead.parents:
        #        parent.children.add(bead)
        #        parent.siblings.update(bead.parents - {parent}) # If I have multiple parents, they must be siblings.

        nchains = 1 # The number of parallel chains
        v = self.vertex(startv)
        if v.in_degree() == 0:
            self.vcolors[v] = genesis_color
        for c in [x.target() for x in v.out_edges()]:
            if c.out_degree() == 0:
                self.vcolors[c] = tip_color
            #else: self.vcolors[c] = bead_color
            if c == highlight:
                for s in [x.target() for x in v.out_edges()]:
                    self.vhcolors[s] = cohort_color
                    if s != c: 
                        self.vcolors[s] = sibling_color
                    # Follow children of siblings until we reach one of my own children
            self.analyze(startv=c)

        #for bead in self.beads: # We really should be following links.
        #    nchains += len(children)-1
        #    children = set(bead.children)
        #    if nchains == 1: self.vcolors["graphcuts"][bead.vertex
        #    for sibling in self.siblings:
                # Find our common ancestor.

        # For each bead:
        #   Traverse all beads between 

    def all_generations(self, v:gt.Vertex, older):
        """ Return the next generation older or younger depending on the value
            of <older>. """
        if older: (edgef, nodef) = ("out_edges","target")
        else:     (edgef, nodef) = ("in_edges", "source")
        result = set()
        for x in [getattr(y, nodef)() for y in getattr(v, edgef)()]:
            result.update({x}.union(self.nextgen(x, older)))
        return result

    def ancestors(self, v:gt.Vertex):
       return self.all_generations(v, older=True)

    def descendants(self, v:gt.Vertex):
       return self.all_generations(v, older=False)

    #def ancestors(self, v:gt.Vertex):
    #    result = set()
    #    for x in [y.source() for y in v.in_edges()]:
    #        result.update({x}.union(self.ancestors(x)))
    #    return result

    #def descendants(self, v:gt.Vertex):
    #    result = set()
    #    for x in [y.target() for y in v.out_edges()]:
    #        result.update({x}.union(self.descendants(x)))
    #    return result

    def next_generation(self, vs, older):
        if older: (edgef, nodef) = ("out_edges","target")
        else:     (edgef, nodef) = ("in_edges", "source")
        if isinstance(vs, gt.Vertex):
            return {getattr(y, nodef)() for y in getattr(vs,edgef)()}
        elif isinstance(vs, set):
            ng = [self.next_generation(v, older) for v in vs]
            if not ng: return set()
            else: return set.union(*ng)

    def parents(self, vs):
        return self.next_generation(vs, older=True)

    def children(self, vs):
        return self.next_generation(vs, older=False)


    #def children(self, vs):
    #    if isinstance(vs, gt.Vertex):
    #        return {y.target() for y in vs.out_edges()}
    #    elif isinstance(vs, set):
    #        result = set()
    #        for v in vs:
    #            result.update(self.children(v))
    #        return result

    #def parents(self, vs):
    #    if isinstance(vs, gt.Vertex):
    #        return {y.source() for y in vs.in_edges()}
    #    elif isinstance(vs, set):
    #        result = set()
    #        for v in vs:
    #            result.update(self.parents(v))
    #        return result

    def highlight(self, vertices=None, hcolor=highlight1_color):
        if not vertices: return
        for v in vertices:
            self.vhcolors[v] = hcolor

    def winners(self, txid):
        """ The "winner" of a transaction <txid> are the beads for which no
            parent includes that txid. """
        pass

    def coinbase(self, cbead):
        """ Allocate coinbase rewards to all parents of <cbead> until we hit
            another coinbase bead. """
        # 1. Starting with <bead>, work up its parent tree building a list of txids.  
        #   a. visit all beads which have not been paid out
        # 2. Re-traverse the list 
        pass

        # If we follow parent links 

        # Starting from the coinbase block, 
        for bead in self.beads:
            # 
            descendents = {}
            ancestors = {}

    def plot(self, focusbead=None, focuscohort=None, numbervertices=False,
            output=None):
        """ Plot this braid, possibly coloring graph cuts.  <focusbead>
            indicates which bead to consider for coloring its siblings and
            cohort. """
        ancestors = set()
        descendants = set()
        vorder = self.new_vertex_property("float")
        kwargs = {'vertex_size':8, 
                  'nodesfirst':True,
                  #vertex_text':numbervertices and self.vertex_index or None, # FIXME use rank instead if focusbead
                  'vertex_halo':True, 
                  'vertex_fill_color':self.vcolors,
                  'vertex_halo_color':self.vhcolors, 
                  'vorder':vorder}
        if output: kwargs['output'] = output
        if focusbead:
            ancestors   = self.ancestors(focusbead)
            descendants = self.descendants(focusbead)
            siblings    = self.siblings(focusbead)
            kwargs['vertex_halo_size'] = 2
            for v in self.vertices():
                # Decide the bead's color
                if v.in_degree() == 0:
                    self.vcolors[v] = genesis_color
                elif v.out_degree() == 0:
                    self.vcolors[v] = tip_color
                elif siblings and v in siblings:
                    self.vcolors[v] = sibling_color
                else:
                    self.vcolors[v] = bead_color
                # Decide the highlight color
                if v == focusbead:
                    self.vhcolors[v] = me_color
                elif v in ancestors:
                    self.vhcolors[v] = ancestor_color
                elif v in descendants:
                    self.vhcolors[v] = descendant_color
                else:
                    self.vhcolors[v] = nohighlight_color
                # Indicate ordering of vertices
                vorder[v] = self.beads[int(v)].t
        else:
            cnum = 0
            #ncohorts = len(list(self.cohorts()))
            for c in self.cohorts():
                for v in c:
                    if focuscohort == cnum:
                        self.vhcolors[v] = highlight1_color
                    self.vcolors[v] = color_palette[cnum%len(color_palette)]
                cnum += 1


        if numbervertices: kwargs['vertex_text']= self.vertex_index
        gtdraw.graph_draw(self, **kwargs)

