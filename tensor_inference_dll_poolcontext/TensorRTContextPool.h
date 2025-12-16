#pragma once
#include <queue>
#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <NvInfer.h>

using namespace nvinfer1;
//主要是防止，多线程反复调用同一个context，会出现报错
class TensorRTContextPool {
public:
    /**
     * @brief 构造函数
     * @param engine TensorRT引擎指针（外部管理生命周期）
     * @param max_contexts 最大上下文数量
     */
    explicit TensorRTContextPool(ICudaEngine* engine, size_t max_contexts = 4);

    /**
     * @brief 从池中获取一个可用的上下文
     * @return 返回上下文的智能指针
     * @throw std::runtime_error 当没有可用上下文时
     */
    std::shared_ptr<IExecutionContext> acquireContext();

    /**
     * @brief 获取可用的上下文数量
     */
    size_t getAvailableContextCount() const;

    /**
     * @brief 获取正在使用的上下文数量
     */
    size_t getInUseContextCount() const;

    /**
     * @brief 获取总上下文数量
     */
    size_t getTotalContextCount() const;
    void notifyContextReleased();
private:
    // TensorRT引擎指针（外部管理生命周期）
    ICudaEngine* engine_;

    // 上下文池
    std::vector<std::shared_ptr<IExecutionContext>> contexts_;

    // 同步原语
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    
    // 配置
    size_t max_contexts_;
};
class ContextPool {
private:
    std::vector<IExecutionContext*> contexts;    // 所有上下文的存储
    std::queue<IExecutionContext*> free_ctx;      // 可用上下文队列
    std::mutex mtx;                               // 互斥锁
    std::condition_variable cv;                   // 条件变量
    bool shutdown_flag = false;                   // 关闭标志

public:
    // 构造函数：预创建指定数量的上下文
    ContextPool(int pool_size, ICudaEngine* engine) {
        if (!engine) {
            throw std::runtime_error("Invalid engine pointer");
        }

        for (int i = 0; i < pool_size; ++i) {
            IExecutionContext* ctx = engine->createExecutionContext();
            if (!ctx) {
                throw std::runtime_error("Failed to create execution context");
            }

            // 动态形状引擎需要设置优化配置文件
            if (!engine->hasImplicitBatchDimension()) {
                if (!ctx->setOptimizationProfile(i % engine->getNbOptimizationProfiles())) {
                    ctx->destroy();
                    throw std::runtime_error("Failed to set optimization profile");
                }
            }

            contexts.push_back(ctx);
            free_ctx.push(ctx);
        }
    }

    // 析构函数：释放所有资源
    ~ContextPool() {
        {
            std::lock_guard<std::mutex> lock(mtx);  //作用域结束，lock 析构，自动解锁
            shutdown_flag = true;  // 设置关闭标志
        }
        cv.notify_all();  // 唤醒所有等待线程

        for (auto* ctx : contexts) {
            if (ctx) {
                ctx->destroy();
            }
        }
    }

    // 获取上下文（阻塞直到可用）
    IExecutionContext* acquire() {
        std::unique_lock<std::mutex> lock(mtx);

        // 等待直到有可用上下文或池已关闭
        cv.wait(lock, [this]() {
            return !free_ctx.empty() || shutdown_flag;
            });

        if (shutdown_flag) {
            return nullptr;  // 池已关闭，返回空指针
        }

        IExecutionContext* ctx = free_ctx.front();
        free_ctx.pop();
        return ctx;
    }

    // 释放上下文回池
    void release(IExecutionContext* ctx) {
        if (!ctx) return;

        {
            std::lock_guard<std::mutex> lock(mtx);
            if (shutdown_flag) return;  // 池已关闭不再接受回收

            free_ctx.push(ctx);
        }
        cv.notify_one();  // 通知一个等待线程
    }

    // 禁止拷贝和赋值
    ContextPool(const ContextPool&) = delete;
    ContextPool& operator=(const ContextPool&) = delete;
};

