from six.moves.urllib.request import urlopen
from six.moves.urllib.error import HTTPError
from six.moves.queue import Queue
import re
from threading import Thread
import sys

def worker(task,poems):
    while True:
        js = task.get()
        fetch_poems(js,poems)
        task.task_done()

def write_poems(path,poems):
    with open(path,'w',encoding="utf-8") as f:
        n = 0
        while True:
            poem = poems.get()
            if not poem:
                break
            f.write(poem)
            f.write('\n')
            poems.task_done()
            n += 1
            sys.stdout.write('\r')
            sys.stdout.write('{} poems have been writen to the file'.format(n))
            sys.stdout.flush()

def fetch_poems(js,poems):
    ns = 1
    while True:
        poem = get_poem(js,ns)
        if not poem:
            break
        poems.put(poem)
        ns += 1

def get_poem(js=1,ns=1):
    try:
        raw_data = urlopen('http://www16.zzu.edu.cn/qtss/zzjpoem1.dll/viewoneshi?js={:03d}&ns={:03d}'.format(js,ns)).read().decode('gb2312','ignore')
    except HTTPError as error:
        print('HTTP Error {}:{}'.format(error.code,error.reason))
        return None
    match = re.search(r'color="#FFFFBF">(.*?)</font>',raw_data,re.M)
    if match:
        return match.group(1).replace('&nbsp;','').replace('<br>','')
    return None

def main():
    volumes = 900
    num_worker_threads = 25
    task = Queue()
    poems = Queue()
    for i in range(num_worker_threads):
        t = Thread(target=worker,args=(task,poems))
        t.daemon = True
        t.start()
    write_thread = Thread(target=write_poems,args=('./data/poems.txt',poems))
    write_thread.start()
    for js in range(1,volumes+1):
        task.put(js)
    task.join()
    poems.join()
    poems.put(None)
    write_thread.join()

if __name__ == "__main__":
    main()
