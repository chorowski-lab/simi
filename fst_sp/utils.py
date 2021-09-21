from collections import defaultdict, namedtuple
import multiprocessing as mp
import math

Sentence = namedtuple('Sentence', ['text', 'count'])
SentencePiece = namedtuple('SentencePiece', ['index', 'symbol', 'log_freq'])
PieceCounts = namedtuple('PieceCounts', ['Z', 'counts'])
ViterbiPath = namedtuple('ViterbiPath', ['path', 'prob', 'log_prob'])
EStepRet = namedtuple('EStepRet', ['objective', 'n_tokens', 'counts'])

def add_dicts(a, b, mult=1):
    for symb, count in b.items():
        a[symb] += count*mult
    return a

def parallelize(
        worker,
        data: list,
        additional_args: list = [],
        aggregating_ops: list = [],
        n_workers = int(max(mp.cpu_count()/2-1, 2))
        ):
    """
    Worker fuction should be of type:
        def worker(data: list, additional_args..., outp: mp.Queue):
            ...
            outp.put([o1,o2,o3,...])
    Paralelize fuction splits the data equally for each worker and run them in separate processes.
    Each worker analize separate subset of data (e.g. data[100:200]). When the work is finished
    result is pushed to queue. Parent process will wait for n_workers items added to queue and agregate
    each of element on the output tuple/array separatelly by fuction given by aggregating_ops i.e.
    If function returns 'res'  and there was 3 workers then:
        res[k] = aggregating_ops[k](aggregating_ops[k](w1[k],w2[k]),w3)
    where w1 is the worker which first finished its job (not always the first chunk of data!), etc.

    *Number of workers should be similar to number of physical cores to best gain in efficiency
    *Function do not care about order of data - so aggregating fuctions should be commutative and associative

    Example:
        def worker(data,arg,outp):
            s = sum(data) if len(data) else 0
            p = max(max(data),arg) if len(data) else -math.inf
            outp.put([s,p])
        summ, maxx= parallelize(worker, 
                            list(range(100)),
                            additional_args=[3],
                            aggregating_ops= [lambda x,y: x+y, lambda x,y: max(x,y)],
                            n_workers = 2)
        assert(summ == sum(range(100)))
        assert(maxx == 99)
        print(summ,maxx)
    """
    chunksize = int(math.ceil(len(data)/n_workers))
    out_q = mp.Queue()
    procs = []

    for i in range(n_workers):
        p = mp.Process(
            target=worker,
            args=(data[chunksize*i:chunksize*(i+1)],
                *additional_args,
                out_q),
        )
        procs.append(p)
        p.start()
    
    acc_res = []
    for i in range(n_workers):
        res = out_q.get()
        if i==0:
            acc_res=res
            continue

        for k,out in enumerate(res):
            acc_res[k] = aggregating_ops[k](acc_res[k],out)

    for p in procs:
        p.join()

    return acc_res

if __name__=='__main__':
    def worker(data,arg,outp):
        s = sum(data) if len(data) else 0
        p = max(max(data),arg) if len(data) else -math.inf
        outp.put([s,p])
    summ, maxx= parallelize(worker, 
                        list(range(100)),
                        additional_args=[3],
                        aggregating_ops= [lambda x,y: x+y, lambda x,y: max(x,y)],
                        n_workers = 2)
    assert(summ == sum(range(100)))
    assert(maxx == 99)
    print(summ,maxx)
    