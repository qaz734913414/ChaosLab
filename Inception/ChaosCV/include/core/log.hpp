#pragma once

#include "core/def.hpp"

#include <string>
#include <sstream>

namespace chaos
{
	class CHAOS_API LogMessage
	{
	public:
		LogMessage(const char* file, int line, LogSeverity severity);
		~LogMessage();

		std::stringstream& Stream();
	private:
		void Flush();

		std::stringstream message_data; // use stringstream to replace LogMessageData
		LogSeverity severity;
	};

	class CHAOS_API LogMessageVoidify
	{
	public:
		LogMessageVoidify() {}
		// This has to be an operator with a precedence lower than << but
		// higher than ?:
		void operator&(std::ostream&) {}
	};

	CHAOS_API void InitLogging(const std::string argv0, int level = 0);
}