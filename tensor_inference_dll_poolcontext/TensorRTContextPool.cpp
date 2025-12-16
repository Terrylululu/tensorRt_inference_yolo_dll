#include "TensorRTContextPool.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;
TensorRTContextPool::TensorRTContextPool(ICudaEngine* engine, size_t max_contexts)
    : engine_(engine)
    , max_contexts_(max_contexts) {

    if (!engine_) {
        throw std::runtime_error("Invalid TensorRT engine");
    }

    if (max_contexts_ == 0) {
        throw std::runtime_error("max_contexts must be greater than 0");
    }

    // 预创建所有上下文
    contexts_.reserve(max_contexts_);
    for (size_t i = 0; i < max_contexts_; ++i) {
        auto context = std::shared_ptr<IExecutionContext>(
            engine_->createExecutionContext(),
            [](IExecutionContext* ctx) {
                if (ctx) ctx->destroy();
            }
        );

        if (!context) {
            throw std::runtime_error("Failed to create TensorRT context");
        }

        contexts_.push_back(context);
    }
}

std::shared_ptr<IExecutionContext> TensorRTContextPool::acquireContext() {
    std::unique_lock<std::mutex> lock(mutex_);

    // 寻找可用的上下文
    for (auto& ctx : contexts_) {
        if (ctx.use_count() == 1) {  // 只有池本身持有引用，说明可用
            return ctx;
        }
    }

    throw std::runtime_error("No available contexts in pool");
}

size_t TensorRTContextPool::getAvailableContextCount() const {
    std::unique_lock<std::mutex> lock(mutex_);

    size_t count = 0;
    for (const auto& ctx : contexts_) {
        if (ctx.use_count() == 1) {
            ++count;
        }
    }

    return count;
}

size_t TensorRTContextPool::getInUseContextCount() const {
    std::unique_lock<std::mutex> lock(mutex_);

    size_t count = 0;
    for (const auto& ctx : contexts_) {
        if (ctx.use_count() > 1) {
            ++count;
        }
    }

    return count;
}

size_t TensorRTContextPool::getTotalContextCount() const {
    return max_contexts_;
}
void TensorRTContextPool::notifyContextReleased() {
    condition_.notify_one();
}
