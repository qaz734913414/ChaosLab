#pragma once

#ifdef CHAOS_EXPORT
#define CHAOS_API __declspec(dllexport)
#else
#define CHAOS_API __declspec(dllimport)
#endif

namespace chaos
{
	enum LogSeverity
	{
		INFO,
		WARNING,
		ERROR,
		FATAL,
	};

	// Pre-declaration
	class LogMessage;
	class LogMessageVoidify;
	class Flag;
	class FlagRegisterer;

} // namespace chaos

using namespace chaos;

#define LOG(severity) LogMessage(__FILE__, __LINE__, severity).Stream()

#define DO_NOT_IMPLEMENTED LOG(FATAL) << "DO NOT IMPLEMENT"

#define CHECK(condition) condition ? (void)0 :			\
  LogMessageVoidify() & LOG(chaos::FATAL) <<			\
  "Check failed: " #condition ". "

#define CHECK_EQ(val1, val2) CHECK(val1 == val2)
#define CHECK_NE(val1, val2) CHECK(val1 != val2)
#define CHECK_LE(val1, val2) CHECK(val1 <= val2)
#define CHECK_LT(val1, val2) CHECK(val1 <  val2)
#define CHECK_GE(val1, val2) CHECK(val1 >= val2)
#define CHECK_GT(val1, val2) CHECK(val1 >  val2)

#define DEFINE_FLAG(type, name, value, group, help)									\
  namespace flag_##type {															\
    static type flag_##name = value;												\
    Flag o_##name(&flag_##name);													\
    FlagRegisterer r_##name(#name, o_##name, #type, #value, help, group, __FILE__); \
  }																					\
  using flag_##type::flag_##name;

#define DEFINE_BOOL(name, group, help)			\
  DEFINE_FLAG(bool, name, false, group, help)

#define DEFINE_INT(name, value, group, help)	\
  DEFINE_FLAG(int, name, value, group, help)

#define DEFINE_FLOAT(name, value, group, help)	\
  DEFINE_FLAG(float, name, value, group, help)

#define DEFINE_STRING(name, value, group, help) \
  using std::string;							\
  DEFINE_FLAG(string, name, value, group, help)