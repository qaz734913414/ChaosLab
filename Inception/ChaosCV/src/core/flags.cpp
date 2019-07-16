#include "core/core.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <set>

namespace chaos
{
	DEFINE_BOOL(help, "", "show help on all flags");

	class FlagInfo
	{
	public:
		bool operator<(const FlagInfo& data) const
		{
			return name < data.name;
		}

		std::string name;
		Flag flag;
		std::string type;
		std::string help;
		std::string value; // default value
		std::string group;
		std::string from; // file from
	};

	class FlagsPool
	{
	public:
		void Help(const std::string& restrict_file = std::string())
		{
			auto ShowInfo = [](const std::set<FlagInfo>& infos) {
				for (auto info : infos)
				{
					std::cout << "    -" << info.name << " (" << info.help << ") type: " << info.type
						<< " default: " << info.value << std::endl;
				}
			};

			for (auto file : from_list)
			{
				if (!restrict_file.empty() && restrict_file != file)
					continue;

				std::cout << std::endl << "Flags from " << file << ":" << std::endl;

				// Save flags info to sub in group
				std::map<std::string, std::set<FlagInfo>> sub;
				auto it = list.begin();
				while (list.end() != (it = std::find_if(it, list.end(), [=](const std::map<std::string, FlagInfo>::value_type& value) { return value.second.from == file; })))
				{
					sub[it->second.group].insert(it->second);
					++it;
				}

				// Always show default group firstly
				ShowInfo(sub["default"]);
				sub.erase("default");

				for (auto data : sub)
				{
					std::cout << std::endl << "  " << data.first << ": " << std::endl;
					ShowInfo(data.second);
				}
			}
		}

		void Register(const std::string& name, const FlagInfo& info)
		{
			CHECK_EQ(list.end(), list.find(name)) << "Flag " << name << " already exists";
			list[name] = info;

			from_list.insert(info.from);
			//group_message[info.group] = ""; // To set the group message empty
		}

		template<class Type>
		void Set(const std::string& name, const Type& value)
		{
			CHECK_NE(list.end(), list.find(name)) << "Unknown flag " << name;
			list[name].flag = value;
		}

		// Singleton
		static std::shared_ptr<FlagsPool> Creat()
		{
			static auto pool = std::shared_ptr<FlagsPool>(new FlagsPool());
			return pool;
		}

		FlagInfo Get(const std::string& name) const
		{
			auto it = list.find(name);
			CHECK_NE(list.end(), it) << "Unknown flag " << name;
			return it->second;
		}

		std::string ShowUsageMessage() const
		{
			return usage_message;
		}

		void SetUsageMessage(const std::string& message)
		{
			usage_message = message;
		}

	private:
		std::map<std::string, FlagInfo> list;
		std::set<std::string> from_list;
		//std::map<std::string, std::string> group_message;

		std::string usage_message;
	};


	FlagRegisterer::FlagRegisterer(const std::string& name, const Flag& flag, const std::string& type, const std::string& value, const std::string& help, const std::string& group, const std::string& file)
	{
		FlagInfo info;

		info.name = name;
		info.flag = flag;
		info.type = type;
		info.help = help;
		info.value = value;
		info.group = group.empty() ? "default" : group;

		File from(file);
		info.from = from.Name + "." + from.Type;

		auto pool = FlagsPool::Creat();
		pool->Register(name, info);
	}

	template<class Type>
	Type ConvertTo(const std::string& data)
	{
		std::stringstream stream;
		stream << data;
		Type value;
		stream >> value;
		return value;
	}

	void ParseCommondLineFlags(int* argc, char*** argv, bool remove_flags)
	{
		auto pool = FlagsPool::Creat();

		std::vector<std::string> rest_argv;
		for (int i = 0; i < *argc; i++)
		{
			if ('-' == (*argv)[i][0]) // First char is '-'
			{
				std::string arg = (*argv)[i];
				std::string name;
				std::string value;
				std::string type;
				if ('-' == (*argv)[i][1]) // Second char is '-'
				{
					auto data = Split(arg, "=");
					// 1 <= data.size() <= 2
					CHECK_GE(data.size(), 1);
					CHECK_LE(data.size(), 2);

					name = data[0].substr(2);
					type = pool->Get(name).type;
					value = data.size() == 2 ? data[1] : (type == "bool" ? "true" : "");
				}
				else
				{
					name = arg.substr(1);
					type = pool->Get(name).type;
					value = type == "bool" ? "true" : (i + 1 < *argc ? (*argv)[++i] : "");
				}

				CHECK(!value.empty()) << "Get a flag without value";
				switch (Hash(type.c_str()))
				{
				case "int"_hash:
					pool->Set(name, ConvertTo<int>(value));
					break;
				case "float"_hash:
					pool->Set(name, ConvertTo<float>(value));
					break;
				case "bool"_hash:
					pool->Set(name, true);
					break;
				case "string"_hash:
					pool->Set(name, value);
					break;
				default:
					LOG(FATAL);
					break;
				}
			}
			else
			{
				rest_argv.push_back((*argv)[i]);
			}
		}

		if (flag_help)
		{
			ShowUsageMessage();
			exit(0);
		}

		if (remove_flags)
		{
			*argc = (int)rest_argv.size();
			for (int i = 0; i < *argc; i++)
			{
				int len = (int)rest_argv[i].size();
				(*argv)[i] = new char[(size_t)len + 1]();
				memcpy((*argv)[i], rest_argv[i].c_str(), len);
			}
		}
	}

	void SetUsageMessage(const std::string& message)
	{
		auto pool = FlagsPool::Creat();
		pool->SetUsageMessage(message);
	}

	void ShowUsageMessage(const std::string& restrict_file)
	{
		auto pool = FlagsPool::Creat();
		std::cout << pool->ShowUsageMessage() << std::endl;
		pool->Help(restrict_file);
	}
}