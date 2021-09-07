#potrzeba zawsze symbolu zakończenia!!
class Node:
    def __init__(self, sub="", children=None):
        self.sub = sub
        self.ch = children or []
 
class SuffixTree:
    def __init__(self, str):
        self.nodes = [Node()]
        for i in range(len(str)):
            self.addSuffix(str[i:])
 
    def addSuffix(self, suf, n=0):
        def getChildNode(parent_nr,child_nr):
            return self.nodes[ self.nodes[parent_nr].ch[child_nr] ]

        def binary_search(l,cond): #find first element on list that fullfill condition; len(n) means there is no such an element
            if len(l)==0 or cond(l[0]):
                return 0
            lower_bound,strict_upper_bound = 0, len(l)
            while strict_upper_bound-lower_bound>1: # finds the last one that do not fullfill
                mid = int((lower_bound+strict_upper_bound)/2)
                if cond(l[mid]):
                    strict_upper_bound = mid
                else:
                    lower_bound = mid
            return lower_bound+1

        #look for first letter
        letter = suf[0]
        
    
        child_of_n = binary_search(self.nodes[n].ch, lambda k: self.nodes[k].sub[0] >= letter)
        
        if child_of_n==len(self.nodes[n].ch) or getChildNode(n,child_of_n).sub[0]>letter: #we need new child of current node 
            self.nodes.append(Node(suf,[]))
            self.nodes[n].ch.insert(child_of_n,len(self.nodes)-1) # the last one node is the next child
            return


        n2=self.nodes[n].ch[child_of_n] #node that beggins with the same prefix
        sub2 = self.nodes[n2].sub
        assert(sub2[0]==suf[0])

        try:
            # find prefix of remaining suffix in common with child
            j = 0
            while j < len(sub2):
                if suf[j] != sub2[j]:
                    # split n2
                    n3 = n2
                    # new node for the part in common
                    n2 = len(self.nodes)
                    self.nodes.append(Node(sub2[:j], [n3]))
                    self.nodes[n3].sub = sub2[j:] # old node loses the part in common
                    self.nodes[n].ch[child_of_n] = n2
                    break # continue down the tree
                j = j + 1

        self.addSuffix(suf[j:],n2)
 
    def visualize(self):
        if len(self.nodes) == 0:
            print( "<empty>")
            return
 
        def f(n, pre):
            children = self.nodes[n].ch
            if len(children) == 0:
                print(  self.nodes[n].sub)
                return
            print(  self.nodes[n].sub)
            for c in children[:-1]:
                print( pre, "+-",end=" ")
                f(c, pre + " | ")
            print( pre, "+-",end=" ")
            f(children[-1], pre + "  ")
 
        f(0, "")



#t.visualize()

def get_vocab(nodenr, tree, prefix=""): #dfs
    res=[]
    node = tree.nodes[nodenr]
    for child in node.ch:
        res += get_vocab(child,tree,prefix+node.sub)
    if( node.ch and len(prefix+node.sub)>1 ):
        res.append(prefix+node.sub)
    return res

if __name__=="__main__":
    t=SuffixTree("abababaabb_cacacacacacacaca_acacacacacac_ca_caca_cacaca_ac_acac_acacac_abcabcabcabcabcab$")
    res=get_vocab(0,t)
    print()
    for x in res:
        print(x,end=" ")
    print()
    print(len(res))