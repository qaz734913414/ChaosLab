#pragma once

#include "core/def.hpp"
#include "core/log.hpp"
#include "core/flags.hpp"

#include <opencv2/opencv.hpp>

namespace chaos
{

	class CHAOS_API File
	{
	public:
		File();
		File(const std::string& file);
		File(const std::string& path, const std::string& name, const std::string& type);

		operator std::string() const;

		CHAOS_API friend std::ostream& operator << (std::ostream& stream, const File& file);

		std::string GetName() const;
		std::string GetPath() const;
		std::string GetType() const;

		__declspec(property(get = GetName)) std::string Name;
		__declspec(property(get = GetPath)) std::string Path;
		__declspec(property(get = GetType)) std::string Type;
	private:
		std::string path;
		std::string name;
		std::string type;
	};


	CHAOS_API std::vector<std::string> Split(const std::string& data, const std::string& delimiter);


	void SetConsoleTextColor(unsigned short color);

	constexpr size_t prime = 0x100000001B3ull;
	constexpr size_t basis = 0xCBF29CE484222325ull;
	constexpr size_t Hash(const char* data, size_t value = basis)
	{
		return *data ? Hash(data + 1, (*data ^ value) * prime) : value;
	}
	constexpr size_t operator "" _hash(const char* data, size_t)
	{
		return Hash(data);
	}
} // namespace chaos