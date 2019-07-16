#pragma once
#include "core/def.hpp"
#include "core/log.hpp"

#include <map>
#include <string>
#include <typeindex>

// This is a simple implementation for gflags

namespace chaos
{
	class CHAOS_API Flag
	{
	public:
		Flag() {}

		template<class Type>
		Flag(Type* ptr) : ptr(ptr), type(typeid(Type)) {}

		template<class Type>
		Flag& operator=(const Type& value)
		{
			CHECK_EQ(type, typeid(Type)) << "Flag type error";
			*(Type*)ptr = value;
			return *this;
		}

	private:
		std::type_index type = typeid(void);
		void* ptr = nullptr;
	};

	class CHAOS_API FlagRegisterer
	{
	public:
		FlagRegisterer(const std::string& name, const Flag& flag, const std::string& type, const std::string& value, const std::string& help, const std::string& group, const std::string& file);
	};

	CHAOS_API void ParseCommondLineFlags(int* argc, char*** argv, bool remove_flags = true);
	CHAOS_API void SetUsageMessage(const std::string& message);
	CHAOS_API void ShowUsageMessage(const std::string& restrict_file = std::string());
}