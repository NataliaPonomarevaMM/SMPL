#ifndef SINGLETON_HPP
#define SINGLETON_HPP

#include "../smpl/Exception.h"

namespace smpl {
    template <class T>
    class Singleton final {
    private:
        static T *m__singleton;
    public:
        Singleton() = delete;
        Singleton(const Singleton<T> &singleton) = delete;
        ~Singleton() = delete;
        Singleton<T> &operator=(const Singleton<T> &singleton) = delete;

        static T *get();
        static void destroy();
    };

    template <class T>
    T *Singleton<T>::m__singleton = nullptr;

    template <class T>
    T *Singleton<T>::get() {
        if (!m__singleton)
            m__singleton = new T;
        return m__singleton;
    }

    template <class T>
    void Singleton<T>::destroy() {
        if (m__singleton)
            delete m__singleton;
    }
} // namespace smpl
#endif // SINGLETON_H