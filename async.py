# 使用协程并发执行只花费4秒
import asyncio
import time

async def do_some_work(x):
    print("Waiting:",x)
    # 等待的过程中,不会影响其他协程的运行
    await asyncio.sleep(x)
    # time.sleep(x)
    return "Done after {}s".format(x)

start = time.time()

coroutine1 = do_some_work(1)
coroutine2 = do_some_work(2)
coroutine3 = do_some_work(4)

tasks = [
    asyncio.ensure_future(coroutine1),
    asyncio.ensure_future(coroutine2),
    asyncio.ensure_future(coroutine3)
]

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.wait(tasks))
# loop.run_until_complete(tasks[0])

for task in tasks:
    print("Task ret:",task.result())

print("Time:",time.time() - start)
'''
Waiting: 1
Waiting: 2
Waiting: 4
Task ret: Done after 1s
Task ret: Done after 2s
Task ret: Done after 4s
Time: 4.0038135051727295
'''
