#include "Python.h"
#include <sys/types.h>
#ifdef __FreeBSD__
#include <sys/extattr.h>
#elif defined(__SUN__) || defined(__sun__) || defined(__sun)
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <alloca.h>
#else
#include <sys/xattr.h>
#endif

#ifdef __FreeBSD__

#define XATTR_COMPAT_ADD_USER_PREFIX 1

/* FreeBSD compatibility API */

/* FreeBSD specifies the namespace separately from the attribute name.
 * Hence when we set names in the "user" namespace, we should
 * strip the "user." prefix off said name.
 *
 * Normally, this would just be an asthetic difference, but recent versions
 * of ZFS now refuse to set on FreeBSD any attribute name starting with the
 * "user." prefix (this is to allow filesystems to be compatible across
 * FreeBSD and Linux systems without ambiguity.)
 *
 * More details here: https://github.com/openzfs/zfs/commit/5c0061345b824eebe7a6578528f873ffcaae1cdd
 */

#define XATTR_XATTR_NOFOLLOW 0x0001
#define XATTR_XATTR_CREATE 0x0002
#define XATTR_XATTR_REPLACE 0x0004
#define XATTR_XATTR_NOSECURITY 0x0008

#define XATTR_CREATE 0x1
#define XATTR_REPLACE 0x2

static const char *strip_user_prefix(const char *name)
{
    char *stripped_name = (char *)name;

    while (!strncmp(stripped_name, "user.", 5)) {
        stripped_name += 5;
    }

    return stripped_name;
}

/* Converts a FreeBSD format attribute list into a NULL terminated list.
 * The first byte is the length of the following attribute.
 * The "user." prefix is added for compatibility with other platforms in the
 * Python interface.
 */
static void convert_bsd_list(char *namebuf, size_t size)
{
    size_t offset = 0;
    while (offset < size) {
        size_t length = (size_t)(unsigned char)namebuf[offset];
        memmove(namebuf + offset, namebuf + offset + 1, length);
        namebuf[offset + length] = '\0';
        offset += length + 1;
    }
}

static ssize_t xattr_getxattr(const char *path, const char *name,
                              void *value, ssize_t size, u_int32_t position,
                              int options)
{
    if (position != 0 ||
        !(options == 0 || options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }

    if (options & XATTR_XATTR_NOFOLLOW) {
        return extattr_get_link(path, EXTATTR_NAMESPACE_USER,
                                strip_user_prefix(name), value, size);
    }
    else {
        return extattr_get_file(path, EXTATTR_NAMESPACE_USER,
                                strip_user_prefix(name), value, size);
    }
}

static ssize_t xattr_setxattr(const char *path, const char *name,
                              void *value, ssize_t size, u_int32_t position,
                              int options)
{
    size_t rv = 0;
    int nofollow;

    if (position != 0) {
        return -1;
    }

    nofollow = options & XATTR_XATTR_NOFOLLOW;
    options &= ~XATTR_XATTR_NOFOLLOW;

    if (options == XATTR_XATTR_CREATE ||
        options == XATTR_XATTR_REPLACE) {

        /* meh. FreeBSD doesn't really have this in its
         * API... Oh well.
         */
    }
    else if (options != 0) {
        return -1;
    }

    if (nofollow) {
        rv = extattr_set_link(path, EXTATTR_NAMESPACE_USER,
                                strip_user_prefix(name), value, size);
    }
    else {
        rv = extattr_set_file(path, EXTATTR_NAMESPACE_USER,
                                strip_user_prefix(name), value, size);
    }

    /* FreeBSD returns the written length on success, not zero. */
    if (rv >= 0) {
        return 0;
    }
    else {
        return rv;
    }
}

static ssize_t xattr_removexattr(const char *path, const char *name,
                                 int options)
{
    if (!(options == 0 ||
          options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }

    if (options & XATTR_XATTR_NOFOLLOW) {
        return extattr_delete_link(path, EXTATTR_NAMESPACE_USER, strip_user_prefix(name));
    }
    else {
        return extattr_delete_file(path, EXTATTR_NAMESPACE_USER, strip_user_prefix(name));
    }
}

static ssize_t xattr_listxattr(const char *path, char *namebuf,
                               size_t size, int options)
{
    ssize_t rv = 0;
    if (!(options == 0 ||
          options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }

    if (options & XATTR_XATTR_NOFOLLOW) {
        rv = extattr_list_link(path, EXTATTR_NAMESPACE_USER, namebuf, size);
    }
    else {
        rv = extattr_list_file(path, EXTATTR_NAMESPACE_USER, namebuf, size);
    }

    if (rv > 0 && namebuf) {
        convert_bsd_list(namebuf, rv);
    }

    return rv;
}

static ssize_t xattr_fgetxattr(int fd, const char *name, void *value,
                               ssize_t size, u_int32_t position, int options)
{
    if (position != 0 ||
        !(options == 0 || options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }

    if (options & XATTR_XATTR_NOFOLLOW) {
        return -1;
    }
    else {
        return extattr_get_fd(fd, EXTATTR_NAMESPACE_USER, strip_user_prefix(name), value, size);
    }
}

static ssize_t xattr_fsetxattr(int fd, const char *name, void *value,
                               ssize_t size, u_int32_t position, int options)
{
    size_t rv = 0;
    int nofollow;

    if (position != 0) {
        return -1;
    }

    nofollow = options & XATTR_XATTR_NOFOLLOW;
    options &= ~XATTR_XATTR_NOFOLLOW;

    if (options == XATTR_XATTR_CREATE ||
        options == XATTR_XATTR_REPLACE) {
        /* FreeBSD noop */
    }
    else if (options != 0) {
        return -1;
    }

    if (nofollow) {
        return -1;
    }
    else {
        rv = extattr_set_fd(fd, EXTATTR_NAMESPACE_USER,
                            strip_user_prefix(name), value, size);
    }

    /* FreeBSD returns the written length on success, not zero. */
    if (rv >= 0) {
        return 0;
    }
    else {
        return rv;
    }
}

static ssize_t xattr_fremovexattr(int fd, const char *name, int options)
{

    if (!(options == 0 ||
          options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }

    if (options & XATTR_XATTR_NOFOLLOW) {
        return -1;
    }
    else {
        return extattr_delete_fd(fd, EXTATTR_NAMESPACE_USER, strip_user_prefix(name));
    }
}


static ssize_t xattr_flistxattr(int fd, char *namebuf, size_t size, int options)
{
    ssize_t rv = 0;

    if (!(options == 0 ||
          options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }

    if (options & XATTR_XATTR_NOFOLLOW) {
        return -1;
    }
    else {
        rv = extattr_list_fd(fd, EXTATTR_NAMESPACE_USER, namebuf, size);
    }

    if (rv > 0 && namebuf) {
        convert_bsd_list(namebuf, rv);
    }

    return rv;
}

#elif defined(__SUN__) || defined(__sun__) || defined(__sun)

/* Solaris 9 and later compatibility API */
#define XATTR_XATTR_NOFOLLOW 0x0001
#define XATTR_XATTR_CREATE 0x0002
#define XATTR_XATTR_REPLACE 0x0004
#define XATTR_XATTR_NOSECURITY 0x0008

#define XATTR_CREATE 0x1
#define XATTR_REPLACE 0x2

#ifndef u_int32_t
#define u_int32_t uint32_t
#endif

static ssize_t xattr_fgetxattr(int fd, const char *name, void *value,
                               ssize_t size, u_int32_t position, int options)
{
    int xfd;
    ssize_t bytes;
    struct stat statbuf;

    /* XXX should check that name does not have / characters in it */
    xfd = openat(fd, name, O_RDONLY | O_XATTR);
    if (xfd == -1) {
    return -1;
    }
    if (lseek(xfd, position, SEEK_SET) == -1) {
    close(xfd);
    return -1;
    }
    if (value == NULL) {
        if (fstat(xfd, &statbuf) == -1) {
        close(xfd);
        return -1;
        }
    close(xfd);
    return statbuf.st_size;
    }
    /* XXX should keep reading until the buffer is exhausted or EOF */
    bytes = read(xfd, value, size);
    close(xfd);
    return bytes;
}

static ssize_t xattr_getxattr(const char *path, const char *name,
                              void *value, ssize_t size, u_int32_t position,
                              int options)
{
    int fd;
    ssize_t bytes;

    if (position != 0 ||
        !(options == 0 || options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }

    fd = open(path,
          O_RDONLY |
          ((options & XATTR_XATTR_NOFOLLOW) ? O_NOFOLLOW : 0));
    if (fd == -1) {
    return -1;
    }
    bytes = xattr_fgetxattr(fd, name, value, size, position, options);
    close(fd);
    return bytes;
}

static ssize_t xattr_fsetxattr(int fd, const char *name, void *value,
                               ssize_t size, u_int32_t position, int options)
{
    int xfd;
    ssize_t bytes = 0;

    /* XXX should check that name does not have / characters in it */
    xfd = openat(fd, name, O_XATTR | O_TRUNC |
         ((options & XATTR_XATTR_CREATE) ? O_EXCL : 0) |
         ((options & XATTR_XATTR_NOFOLLOW) ? O_NOFOLLOW : 0) |
         ((options & XATTR_XATTR_REPLACE) ? O_RDWR : O_WRONLY|O_CREAT),
         0644);
    if (xfd == -1) {
    return -1;
    }
    while (size > 0) {
    bytes = write(xfd, value, size);
    if (bytes == -1) {
        close(xfd);
        return -1;
    }
    size -= bytes;
    value += bytes;
    }
    close(xfd);
    return 0;
}

static ssize_t xattr_setxattr(const char *path, const char *name,
                              void *value, ssize_t size, u_int32_t position,
                              int options)
{
    int fd;
    ssize_t bytes;

    if (position != 0) {
        return -1;
    }

    fd = open(path,
          O_RDONLY | (options & XATTR_XATTR_NOFOLLOW) ? O_NOFOLLOW : 0);
    if (fd == -1) {
    return -1;
    }
    bytes = xattr_fsetxattr(fd, name, value, size, position, options);
    close(fd);
    return bytes;
}

static ssize_t xattr_fremovexattr(int fd, const char *name, int options)
{
  int xfd, status;
    /* XXX should check that name does not have / characters in it */
    if (!(options == 0 || options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }
    if (options & XATTR_XATTR_NOFOLLOW) {
        return -1;
    }
    xfd = openat(fd, ".", O_XATTR, 0644);
    if (xfd == -1) {
    return -1;
    }
    status = unlinkat(xfd, name, 0);
    close(xfd);
    return status;
}

static ssize_t xattr_removexattr(const char *path, const char *name,
                                 int options)
{
    int fd;
    ssize_t status;

    fd = open(path,
          O_RDONLY | ((options & XATTR_XATTR_NOFOLLOW) ? O_NOFOLLOW : 0));
    if (fd == -1) {
    return -1;
    }
    status =  xattr_fremovexattr(fd, name, options);
    close(fd);
    return status;
}

static ssize_t xattr_xflistxattr(int xfd, char *namebuf, size_t size, int options)
{
    int esize;
    DIR *dirp;
    struct dirent *entry;
    ssize_t nsize = 0;

    dirp = fdopendir(xfd);
    if (dirp == NULL) {
        return (-1);
    }
    while (entry = readdir(dirp)) {
        if (strcmp(entry->d_name, ".") == 0 ||
                strcmp(entry->d_name, "..") == 0)
            continue;
        esize = strlen(entry->d_name);
        if (nsize + esize + 1 <= size) {
            snprintf((char *)(namebuf + nsize), esize + 1,
                    entry->d_name);
        }
        nsize += esize + 1; /* +1 for \0 */
    }
    closedir(dirp);
    return nsize;
}
static ssize_t xattr_flistxattr(int fd, char *namebuf, size_t size, int options)
{
    int xfd;

    xfd = openat(fd, ".", O_RDONLY | O_XATTR);
    return xattr_xflistxattr(xfd, namebuf, size, options);
}

static ssize_t xattr_listxattr(const char *path, char *namebuf,
                               size_t size, int options)
{
    int xfd;

    xfd = attropen(path, ".", O_RDONLY);
    return xattr_xflistxattr(xfd, namebuf, size, options);
}

#elif !defined(XATTR_NOFOLLOW)
/* Linux compatibility API */
#define XATTR_XATTR_NOFOLLOW 0x0001
#define XATTR_XATTR_CREATE 0x0002
#define XATTR_XATTR_REPLACE 0x0004
#define XATTR_XATTR_NOSECURITY 0x0008
static ssize_t xattr_getxattr(const char *path, const char *name, void *value, ssize_t size, u_int32_t position, int options) {
    if (position != 0 || !(options == 0 || options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }
    if (options & XATTR_XATTR_NOFOLLOW) {
        return lgetxattr(path, name, value, size);
    } else {
        return getxattr(path, name, value, size);
    }
}

static ssize_t xattr_setxattr(const char *path, const char *name, void *value, ssize_t size, u_int32_t position, int options) {
    int nofollow;
    if (position != 0) {
        return -1;
    }
    nofollow = options & XATTR_XATTR_NOFOLLOW;
    options &= ~XATTR_XATTR_NOFOLLOW;
    if (options == XATTR_XATTR_CREATE) {
        options = XATTR_CREATE;
    } else if (options == XATTR_XATTR_REPLACE) {
        options = XATTR_REPLACE;
    } else if (options != 0) {
        return -1;
    }
    if (nofollow) {
        return lsetxattr(path, name, value, size, options);
    } else {
        return setxattr(path, name, value, size, options);
    }
}

static ssize_t xattr_removexattr(const char *path, const char *name, int options) {
    if (!(options == 0 || options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }
    if (options & XATTR_XATTR_NOFOLLOW) {
        return lremovexattr(path, name);
    } else {
        return removexattr(path, name);
    }
}


static ssize_t xattr_listxattr(const char *path, char *namebuf, size_t size, int options) {
    if (!(options == 0 || options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }
    if (options & XATTR_XATTR_NOFOLLOW) {
        return llistxattr(path, namebuf, size);
    } else {
        return listxattr(path, namebuf, size);
    }
}

static ssize_t xattr_fgetxattr(int fd, const char *name, void *value, ssize_t size, u_int32_t position, int options) {
    if (position != 0 || !(options == 0 || options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }
    if (options & XATTR_XATTR_NOFOLLOW) {
        return -1;
    } else {
        return fgetxattr(fd, name, value, size);
    }
}

static ssize_t xattr_fsetxattr(int fd, const char *name, void *value, ssize_t size, u_int32_t position, int options) {
    int nofollow;
    if (position != 0) {
        return -1;
    }
    nofollow = options & XATTR_XATTR_NOFOLLOW;
    options &= ~XATTR_XATTR_NOFOLLOW;
    if (options == XATTR_XATTR_CREATE) {
        options = XATTR_CREATE;
    } else if (options == XATTR_XATTR_REPLACE) {
        options = XATTR_REPLACE;
    } else if (options != 0) {
        return -1;
    }
    if (nofollow) {
        return -1;
    } else {
        return fsetxattr(fd, name, value, size, options);
    }
}

static ssize_t xattr_fremovexattr(int fd, const char *name, int options) {
    if (!(options == 0 || options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }
    if (options & XATTR_XATTR_NOFOLLOW) {
        return -1;
    } else {
        return fremovexattr(fd, name);
    }
}


static ssize_t xattr_flistxattr(int fd, char *namebuf, size_t size, int options) {
    if (!(options == 0 || options == XATTR_XATTR_NOFOLLOW)) {
        return -1;
    }
    if (options & XATTR_XATTR_NOFOLLOW) {
        return -1;
    } else {
        return flistxattr(fd, namebuf, size);
    }
}

#else /* Mac OS X assumed */
#define xattr_getxattr getxattr
#define xattr_fgetxattr fgetxattr
#define xattr_removexattr removexattr
#define xattr_fremovexattr fremovexattr
#define xattr_setxattr setxattr
#define xattr_fsetxattr fsetxattr
#define xattr_listxattr listxattr
#define xattr_flistxattr flistxattr

/* define these for use in python (see below) */
#define XATTR_XATTR_NOFOLLOW	XATTR_NOFOLLOW
#define XATTR_XATTR_CREATE	XATTR_CREATE
#define XATTR_XATTR_REPLACE	XATTR_REPLACE
#define XATTR_XATTR_NOSECURITY	XATTR_NOSECURITY
#endif

#ifndef XATTR_MAXNAMELEN
#define XATTR_MAXNAMELEN 127
#endif

#ifndef XATTR_COMPAT_ADD_USER_PREFIX
#define XATTR_COMPAT_ADD_USER_PREFIX 0
#endif
