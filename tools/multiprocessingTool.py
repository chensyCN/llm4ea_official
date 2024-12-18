import multiprocessing as mp
import sys
from itertools import chain
from threading import Thread

from tqdm import tqdm

from tools.announce import Announce


class MultiprocessingTool:
    def __init__(self, num_workers=10, use_semaphore=False):
        self._num_workers = num_workers
        self.__sender_queue = mp.Queue()
        self.__result_queue = mp.Queue()
        self.__processes = []
        self.__pack_count = 0
        self._use_semaphore = use_semaphore
        self.invoke_freq = 100
        self.__sem = mp.Semaphore(2000)
        self._min_process_count = 500000
        self._verbose = False

        # self.tqdm_pack_solver = tqdm(desc='solved packages')
        # self.tqdm_receiver = tqdm(desc='received packages')

    def send_packs(self, iterator, pack_size=5000):
        self.tqdm_receiver = tqdm(desc='received packages', file=sys.stdout)
        pack = [None] * pack_size
        i = 0
        j = 0
        for item in iterator:
            if i == pack_size:
                self.__sender_queue.put((j, pack))
                if self._verbose and j % self.invoke_freq == 0:
                    print(' '.join((Announce.printMessage(), 'Pack', str(j), 'sent')))
                pack = [None] * pack_size
                i = 0
                j += 1
            pack[i] = item
            i += 1
        pack = [item for item in pack if item is not None]
        self.__sender_queue.put((j, pack))
        self.__pack_count = j + 1
        # self.tqdm_pack_solver.total = self.__pack_count
        self.tqdm_receiver.total = self.__pack_count
        for i in range(self._num_workers):
            self.__sender_queue.put((None, None))
        print(' '.join((Announce.printMessage(), 'Finished sending packs')))
        print(' '.join((Announce.printMessage(), 'Total Pack Count:', str(self.__pack_count))))
        return self

    def packed_solver(self, item_solver, **kwargs):
        for i in range(self._num_workers):
            p = mp.Process(target=self._packed_solver, args=(item_solver,), kwargs=kwargs)
            self.__processes.append(p)
            p.start()
        return self

    def _packed_solver(self, item_solver, **kwargs):
        while True:
            if self._use_semaphore:
                self.__sem.acquire()
            id, pack = self.__sender_queue.get()
            if id is None:
                break
            result_pack = [None] * len(pack)
            for i, item in enumerate(pack):
                result = item_solver(item, **kwargs)
                result_pack[i] = result
            self.__result_queue.put((id, result_pack))
            if self._verbose:
                print(' '.join((Announce.printMessage(), 'Pack', str(id), 'finished')))
            # self.tqdm_pack_solver.update()
            # sleep(1)
        self.__result_queue.put((None, None))

    def receive_results(self, processor=None, **kwargs):
        finished_workers = 0
        result_packs = [None] * self.__pack_count
        while True:
            id, result_pack = self.__result_queue.get()
            if id is None:
                finished_workers += 1
                if finished_workers == self._num_workers:
                    break
                continue
            try:
                if processor is not None:
                    result_pack = processor(result_pack, **kwargs)
                result_packs[id] = result_pack
            except IndexError:
                print('err', id, len(result_packs))

            if self._verbose:
                print(' '.join((Announce.printMessage(), 'Result', str(id), 'received')))
            self.tqdm_receiver.update()
        self.tqdm_receiver.close()
        return self._results_unpack(result_packs)

    def _results_unpack(self, result_packs):
        # results = list(_flatten(result_packs))
        results = list(chain.from_iterable(result_packs))
        print(Announce.printMessage(), 'concat finished')
        return results

    def reveive_and_process(self, processor, **kwargs):
        finished_workers = 0
        tp = None
        old_id = None
        old_pack = None
        first = True
        multi_pack = None
        count = 0
        pack_count = 0
        while True:
            id, result_pack = self.__result_queue.get()
            if self._use_semaphore:
                self.__sem.release()
            # print(result_pack)
            if id is None:
                finished_workers += 1
                if finished_workers == self._num_workers:
                    break
                continue
            pack_count += 1
            self.tqdm_receiver.update()
            if multi_pack is None:
                multi_pack = result_pack
            else:
                # multi_pack: list
                multi_pack.extend(result_pack)
            c = map(lambda x: len(x), result_pack)
            d = sum(c)
            count += d
            if self._verbose:
                print(' '.join((Announce.printMessage(), 'Result', str(id), str(pack_count), 'get', 'count:', str(count))))
            if count < self._min_process_count:
                continue
            # print(' '.join((Announce.printMessage(), str(id), str(first))))
            if not first:
                tp.join()
                if self._verbose:
                    print(' '.join((Announce.printMessage(), 'Result', str(old_id), 'finished')))
            if first:
                first = False
            # processor(result_pack, **kwargs)
            old_id = id
            old_pack = multi_pack
            print(Announce.printMessage(), 'processing', id)
            tp = Thread(target=processor, args=(old_pack,), kwargs=kwargs)
            tp.start()
            multi_pack = None
            count = 0
        if tp is not None:
            tp.join()
        if self._use_semaphore:
            self.__sem.release()
        print(len(multi_pack))
        if multi_pack is not None:
            print('run')
            old_pack = multi_pack
            tp = Thread(target=processor, args=(old_pack,), kwargs=kwargs)
            tp.start()
            tp.join()
        print(' '.join((Announce.printMessage(), 'Result', str(old_id), 'finished')))
        self.tqdm_receiver.close()
        print(Announce.printMessage(), 'all packs finished')
        pass


MPTool = MultiprocessingTool()

if __name__ == '__main__':
    tool = MultiprocessingTool(num_workers=2)
    l = [i for i in range(500)]
    tool.packed_solver(lambda x, y: x * y, y=2)
    tool.send_packs(l, 5)
    results = tool.receive_results()
    # tool.reveive_and_process(lambda x: print(x))
    # print(results)
