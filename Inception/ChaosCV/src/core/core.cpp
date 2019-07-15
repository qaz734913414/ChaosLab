#include "core/core.hpp"

#include <regex>

namespace chaos
{
	std::vector<std::string> Split(const std::string& data, const std::string& delimiter)
	{
		std::regex regex{ delimiter };
		return std::vector<std::string> {
			std::sregex_token_iterator(data.begin(), data.end(), regex, -1),
				std::sregex_token_iterator()
		};
	}
	
} // namespace chaos

#include <Windows.h>

void chaos::SetConsoleTextColor(unsigned short color)
{
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color | FOREGROUND_INTENSITY);
}