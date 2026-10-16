#pragma once
// common: thread pool (D-CON-01, D-MOD-03). This Phase 1 version is a plain
// std::thread pool with a mutex-guarded queue -- correct and enough for
// Phase 1/2's needs. D-CON-01's work-stealing pool (task graph, P-core
// sizing) is Phase 2 scope; swap the implementation behind this same
// interface without touching call sites.
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace lattice {

class ThreadPool {
public:
    explicit ThreadPool(unsigned num_threads) {
        if (num_threads == 0) num_threads = 1;
        workers_.reserve(num_threads);
        for (unsigned i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template <typename Fn>
    auto submit(Fn&& fn) -> std::future<std::invoke_result_t<Fn>> {
        using R = std::invoke_result_t<Fn>;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<Fn>(fn));
        std::future<R> fut = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    // Runs `task(i)` for i in [0, n_tasks) across the pool, blocking until
    // all complete (D-CON-05's per-segment fan-out shape).
    void fan_out(std::size_t n_tasks, const std::function<void(std::size_t)>& task) {
        std::vector<std::future<void>> futures;
        futures.reserve(n_tasks);
        for (std::size_t i = 0; i < n_tasks; ++i) {
            futures.push_back(submit([&task, i] { task(i); }));
        }
        // get(), not wait(): an exception thrown by a task must reach the caller, not vanish.
        for (auto& f : futures) f.get();
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) return;
                job = std::move(tasks_.front());
                tasks_.pop();
            }
            job();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
};

}  // namespace lattice
