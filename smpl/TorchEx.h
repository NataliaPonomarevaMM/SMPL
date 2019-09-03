#ifndef TORCH_EXTRA_H
#define TORCH_EXTRA_H

#include <torch/torch.h>
#include "exception.h"

namespace smpl {
    class TorchEx final {
    private:
        TorchEx() = delete;
        TorchEx(const TorchEx &TorchEx) = delete;
        ~TorchEx() = delete;
        TorchEx &operator=(const TorchEx &TorchEx) = delete;

        // %% Extended Libraries %%
        template <class Array, class ... Rest>
        static torch::Tensor indexing_impl(torch::Tensor &self, int64_t layer, Array index, Rest ... indices);

        template <class Array>
        static torch::Tensor indexing_impl(torch::Tensor &self, int64_t layer, Array index);
    public:
        // %% Extended Libraries %%
        template <class ... Arrays>
        static torch::Tensor indexing(torch::Tensor &self, Arrays ... indices);
    };

    template <class ... Arrays>
    torch::Tensor TorchEx::indexing(torch::Tensor &self, Arrays ... indices) {
        torch::Tensor out = indexing_impl(self, 0, indices ...);
        return out;
    }

    template <class Array, class ... Rest>
    torch::Tensor TorchEx::indexing_impl(torch::Tensor &self, int64_t layer, Array index, Rest ... indices) {
        if (!std::is_same<Array, torch::IntList>::value)
            throw smpl_error("TorchEx", "Integer list type mismatch in recursion!");

        torch::Tensor slice;
        torch::Tensor out;
        switch (index.size()) {
            case 0:
                slice = self.slice(layer);
                out = indexing_impl(slice, layer + 1, indices ...);
                break;
            case 1:
                slice = self.slice(layer, *(index.begin()), *(index.begin()) + 1).squeeze(layer);
                out = indexing_impl(slice, layer, indices ...);
                break;
            case 2:
                slice = self.slice(layer, *(index.begin()), *(index.begin() + 1));
                out = indexing_impl(slice, layer + 1, indices ...);
                break;
            case 3:
                slice = self.slice(layer, *(index.begin()), *(index.begin() + 1), *(index.begin() + 2));
                out = indexing_impl(slice, layer + 1, indices ...);
                break;
            default:
                throw smpl_error("TorchEx", "Invalid integer list for recursive indexing!");
        }
        return out;
    }

    template <class Array>
    torch::Tensor TorchEx::indexing_impl(torch::Tensor &self, int64_t layer, Array index) {
        if (!std::is_same<Array, torch::IntList>::value)
            throw smpl_error("TorchEx", "Integer list type mismatch in base!");

        torch::Tensor out;
        switch (index.size()) {
            case 0:
                out = self.slice(layer);
                break;
            case 1:
                out = self.slice(layer, *(index.begin()), *(index.begin()) + 1).squeeze(layer);
                break;
            case 2:
                out = self.slice(layer, *(index.begin()), *(index.begin() + 1));
                break;
            case 3:
                out = self.slice(layer, *(index.begin()), *(index.begin() + 1), *(index.begin() + 2));
                break;
            default:
                throw smpl_error("TorchEx", "Invalid integer list for basic indexing!");
        }
        return out;
    }
}
#endif // TORCH_EXTRA_H