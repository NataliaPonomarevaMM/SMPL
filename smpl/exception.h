#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <exception>
#include <string>
#include <sstream>

namespace smpl {
#ifndef smpl_error
#define smpl_error(module, error) Exception(module, error, __func__, __FILE__, __LINE__)
#endif

    class Exception final : public std::exception {
    private:
        std::string m__module;
        std::string m__error;
        std::string m__function;
        std::string m__file;
        int m__line;

        std::stringstream m__stream;
        std::string m__message;
    public:
        // %% Constructor and Destructor %%
        Exception(const std::string module, const std::string error,
                  const std::string function, const std::string file, const int line);
        Exception(const Exception &exception);
        ~Exception() final = default;

        // %% Operators %%
        Exception &operator=(const Exception &exception);

        // %% Exception propagation %%
        const char *what() const final;
    };
} // namespace SMPL
#endif // EXCEPTION_H