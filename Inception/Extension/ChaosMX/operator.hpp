#pragma once

#include "base.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API OpInfo
		{
		public:
			OpInfo();
			OpInfo(const std::string& description, const std::string& key_var_num_args);

			std::string description;
			std::vector<const char*> arg_names;
			std::vector<const char*> arg_type_infos;
			std::vector<const char*> arg_descriptions;
			std::string key_var_num_args;
		};

		class CHAOS_API OpMapper
		{
		public:
			OpMapper();

			AtomicSymbolCreator GetCreator(const std::string& op_name) const;
			OpInfo GetInfo(const std::string& op_name) const;
		private:
			std::map<std::string, AtomicSymbolCreator> op_creators;
			std::map<std::string, OpInfo> op_infos;
		};

		class CHAOS_API Operator
		{
		public:
			Operator();
			Operator(const std::string& op_name);

			std::string op_name;
			OpInfo op_info;
			AtomicSymbolCreator creator;

		private:
			static const Ptr<OpMapper> mapper;
		};

		
	}
}