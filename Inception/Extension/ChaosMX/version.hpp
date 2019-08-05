#pragma once

#define MAJOR_VERSION 1
#define MINOR_VERSION 0
#define REVISION 0
#define BUILD 51 

#define _T(val) #val
#define __T(val) _T(val)

#define RELEASE_VER MAJOR_VERSION, MINOR_VERSION, REVISION, BUILD
#define RELEASE_VER_STR __T(MAJOR_VERSION) "." __T(MINOR_VERSION) "." __T(REVISION) "." __T(BUILD)
