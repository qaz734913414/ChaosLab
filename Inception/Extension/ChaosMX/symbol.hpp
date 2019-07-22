#pragma once

#include "base.hpp"
#include "utils/json.hpp"

#include "operator.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Inputs
		{
		public:
			Inputs();
			Inputs(const Json& json);

			int node_id;
			int index;
			int version;
		};

		class CHAOS_API Symbol
		{
		public:
			using Attrs = std::map<std::string, std::string>;

			Symbol();
			Symbol(const Json& json);

			Operator op;
			std::string name;
			Attrs attrs;
			std::vector<Inputs> inputs;
		};
	}
}