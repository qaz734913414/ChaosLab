#include "operator.hpp"

namespace chaos
{
	namespace dnn
	{


		OpInfo::OpInfo() {}
		OpInfo::OpInfo(const std::string& description, const std::string& key_var_num_args) : description(description), key_var_num_args(key_var_num_args)
		{
		}

		OpMapper::OpMapper()
		{
			mx_uint num_symbol_creators = 0;
			AtomicSymbolCreator* creators = nullptr;
			CHECK_EQ(0, MXSymbolListAtomicSymbolCreators(&num_symbol_creators, &creators)) << MXGetLastError();
			for (mx_uint i = 0; i < num_symbol_creators; i++)
			{
				const char* name;
				
				const char* description = nullptr;
				mx_uint num_args = 0;
				const char** arg_names = nullptr;
				const char** arg_type_infos = nullptr;
				const char** arg_descriptions = nullptr;
				const char* key_var_num_args = nullptr;

				CHECK_EQ(0, MXSymbolGetAtomicSymbolInfo(creators[i], &name, 
					&description, &num_args, &arg_names, 
					&arg_type_infos, &arg_descriptions, &key_var_num_args)) << MXGetLastError();

				op_creators[name] = creators[i];
				op_infos[name] = OpInfo(description, key_var_num_args);
				for (mx_uint i = 0; i < num_args; i++)
				{
					op_infos[name].arg_names.push_back(arg_names[i]);
					op_infos[name].arg_type_infos.push_back(arg_type_infos[i]);
					op_infos[name].arg_descriptions.push_back(arg_descriptions[i]);
				}
			}
		}
		AtomicSymbolCreator OpMapper::GetCreator(const std::string& op_name) const
		{
			auto it = op_creators.find(op_name);
			CHECK_NE(it, op_creators.end()) << "Unsupported op " << op_name;
			return it->second;
		}
		OpInfo OpMapper::GetInfo(const std::string& op_name) const
		{
			auto it = op_infos.find(op_name);
			CHECK_NE(it, op_infos.end()) << "Unsupported op " << op_name;
			return it->second;
		}


		const Ptr<OpMapper> Operator::mapper = Ptr<OpMapper>(new OpMapper());
		Operator::Operator() : op_name("null"), creator(nullptr), op_info(OpInfo()) {}
		Operator::Operator(const std::string& op_name) : op_name(op_name)
		{
			if (op_name == "null")
			{
				creator = nullptr;
				op_info = OpInfo();
			}
			else
			{
				creator = mapper->GetCreator(op_name);
				op_info = mapper->GetInfo(op_name);
			}
		}
	}
}