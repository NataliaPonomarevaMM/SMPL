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
        Exception::Exception(const std::string module, const std::string error,
                             const std::string function, const std::string file, const int line) :
                m__module(module),
                m__error(error),
                m__function(function),
                m__file(file),
                m__line(line)
        {
            m__stream << m__module << " Error: ";
            m__stream << m__error << std::endl;
            m__stream << "Broken Function: " << m__function << std::endl;
            m__stream << "Broken File: " << m__file << std::endl;
            m__stream << "Broken Line: " << m__line << std::endl;

            m__message = m__stream.str();
        }
        Exception::Exception(const Exception &exception) {
            *this = exception;
        }
        ~Exception() final = default;

        // %% Operators %%
        Exception &Exception::operator=(const Exception &exception) {
            m__module = exception.m__module;
            m__error = exception.m__error;
            m__function = exception.m__function;
            m__file = exception.m__file;
            m__line = exception.m__line;

            m__stream << m__module << " Error: ";
            m__stream << m__error << std::endl;
            m__stream << "Broken Function: " << m__function << std::endl;
            m__stream << "Broken File: " << m__file << std::endl;
            m__stream << "Broken Line: " << m__line;

            m__message = m__stream.str();

            return *this;
        }
        // %% Exception propagation %%
        const char *Exception::what() const noexcept {
            return m__message.c_str();
        }
    };
} // namespace SMPL
#endif // EXCEPTION_H