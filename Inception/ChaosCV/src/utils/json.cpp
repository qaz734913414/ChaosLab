#include "utils/json.hpp"

#include <stack>

namespace chaos
{
	Json::Json() {}
	Json::Json(const std::string& json)
	{
		auto GetJsonData = [=]() {
			std::string j;
			for (auto c : json.substr(1, json.size() - 2))
			{
				if (c != '\\') j.push_back(c);
			}
			return j;
		};

		switch (json[0])
		{
		case '{': // Json object
		{
			CHECK_EQ(json.back(), '}');
			std::string j = GetJsonData();

			std::stack<char> bracket;
			while (!j.empty())
			{
				size_t pos = j.find_first_of(":");
				std::string key = j.substr(1, pos - 2);
				j = j.substr(pos + 1);

				std::string val;
				for (int i = 0; i < (int)j.size(); i++)
				{
					if (j[i] == ',' && bracket.empty())
					{
						break;
					}

					val.push_back(j[i]);

					if (j[i] == '(' || j[i] == '[' || j[i] == '{') bracket.push(j[i]);
					if (j[i] == ')' || j[i] == ']' || j[i] == '}') bracket.pop();
				}

				j = j.substr(std::min(j.size(), val.size() + 1));

				if (val.front() == '"' && val.back() == '"') val = val.substr(1, val.size() - 2);
				data[key] = val;
			}
			break;
		}
		case '[': // Json array
		{
			CHECK_EQ(json.back(), ']');
			std::string j = GetJsonData();

			std::stack<char> bracket;
			size_t idx = 0;
			while (!j.empty())
			{
				std::string val;
				for (int i = 0; i < (int)j.size(); i++)
				{
					val.push_back(j[i]);

					if (j[i] == '(' || j[i] == '[' || j[i] == '{') bracket.push(j[i]);
					if (j[i] == ')' || j[i] == ']' || j[i] == '}') bracket.pop();

					if (j[i] == ',' && bracket.empty())
					{
						val.pop_back();
						break;
					}
				}
				data[std::to_string(idx++)] = val;

				j = j.substr(std::min(j.size(), val.size() + 1));
			}
			break;
		}
		default:
			break;
		}
	}

	Json Json::operator[](const std::string& key) const
	{
		auto it = data.find(key);
		return it != data.end() ? it->second : ""; //data.find(key)->second;
		//return data[key];
	}
	Json Json::operator[](const size_t idx) const
	{
		auto it = data.find(std::to_string(idx));
		return it != data.end() ? it->second : ""; // data.find(std::to_string(idx))->second;
		//return data[std::to_string(idx)];
	}
	std::map<std::string, std::string> Json::GetData() const
	{
		return data;
	}
}