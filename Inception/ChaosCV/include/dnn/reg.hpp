#pragma once

#include "dnn/net.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Registered
		{
		public:
			static std::vector<Framework> frameworks;
			static Framework& Have(const std::string& name);
		};

		/// <summary>
		/// <para>Register the framework</para>
		/// <para>For a new framework, you must implement the class inherit class Net</para>
		/// <para>and Load function for new framework, then use marco REGISTER_FRAMEWORK</para>
		/// <para>to register the new framework.</para>
		/// </summary>
		/// <param name="name">framework name</param>
		CHAOS_API Framework& Register(const std::string& name);
	} // namespace dnn
} // namespace chaos

/// <summary>
/// <para>Register a dnn framework for class Net</para>
/// <para>@param name: Framework name</para>
/// <para>@param stype: Symbol file type</para>
/// <para>@param wtype: Weight file type</para>
/// <para>@param func: Load function</para>
/// </summary>
#define REGISTER_FRAMEWORK(name, stype, wtype, func)	\
  namespace _register_##name {							\
    auto _##name = chaos::dnn::Register(#name)			\
      .With(##stype, ##wtype, func);					\
  }

#ifdef USE_MXNET
REGISTER_FRAMEWORK(MxNet, "json", "params", chaos::dnn::LoadMxNet);
#endif