#pragma once

#include "core/core.hpp"

namespace chaos
{
	class CHAOS_API Json
	{
	public:
		Json();
		/// <summary>The json string must be shrinked</summary>
		Json(const std::string& json);

		Json operator[](const std::string& key);
		Json operator[](const int idx);

		std::map<std::string, std::string> GetData() const;
		__declspec(property(get = GetData)) std::map<std::string, std::string> Data;
	private:
		std::map<std::string, std::string> data;
	};
}