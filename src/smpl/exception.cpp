#include "../../include/smpl/exception.h"

namespace smpl {
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

    const char *Exception::what() const noexcept {
        return m__message.c_str();
    }
} // namespace SMPL