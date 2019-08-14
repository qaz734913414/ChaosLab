#pragma once

#include "core/def.hpp"
#include "core/log.hpp"
#include "core/flags.hpp"

#include <opencv2/opencv.hpp>

#include <any>
#include <memory>

namespace chaos
{
	template<class Type>
	using Ptr = std::shared_ptr<Type>;

	using uchar = unsigned char;
	using uint = unsigned int;

	using Mat = cv::Mat;
	using Point = cv::Point2f;
	using Size = cv::Size2f;
	using Rect = cv::Rect2f;
	using Scalar = cv::Scalar;
	// cv::Range do not support float
	using Range = cv::Vec2f; // <min, max>

	class CHAOS_API Shape
	{
	public:
		Shape();

		template<class Type>
		Shape(const std::vector<Type>& data)
		{
			for (auto val : data)
			{
				shape.push_back((int)val);
			}
		}

		template<class Type>
		Shape(const std::initializer_list<Type>& list)
		{
			for (auto val : list)
			{
				shape.push_back((int)val);
			}
		}

		template<class Type>
		operator std::vector<Type>() const
		{
			std::vector<Type> ret;
			for (auto val : shape)
			{
				ret.push_back((Type)val);
			}
			return ret;
		}

		void Swap(Shape& shape);
		size_t Size() const;
		const int& operator[](size_t idx) const;
		int& operator[](size_t idx);

		CHAOS_API friend inline bool operator==(const Shape& s1, const Shape& s2)
		{
			return s1.shape == s2.shape;
		}
		CHAOS_API friend inline std::ostream& operator<<(std::ostream& stream, const Shape& shape);

		std::string ToString() const;


		const int* data() const;
		std::vector<int>::const_iterator begin() const;
		std::vector<int>::iterator begin();
		std::vector<int>::const_iterator end() const;
		std::vector<int>::iterator end();
		int& back();
		const int& back() const;

	private:
		std::vector<int> shape;
	};

	/// <summary>Base class for those which need indefinite parameters</summary>
	class CHAOS_API IndefiniteParameter
	{
	public:
		/// <summary>To set args</summary>
		template<class ... Args>
		void Set(const Args& ... args)
		{
			DummyWrap(Unpack(args)...);
		}

	protected:
		virtual void Parse(const std::any& any) {}

		template <class ... Args>
		void DummyWrap(const Args& ... args) {}

		template <class Arg>
		Arg& Unpack(Arg& arg)
		{
			std::any any = arg;
			Parse(any);
			return arg;
		}

		std::any arg_value;
	};

	class CHAOS_API ProgressBar
	{
	public:
		static void Render(const std::string& message, size_t total = 0, int len = 20);
		static void Update(int step = 1);
		static void Halt();

	private:
		static void Refresh(const std::string& message, size_t total, const int len);

		static bool running;
		static bool stopped;
		static char* progress;
		static size_t current;
		static std::mutex mtx;
		static std::condition_variable halted;
	};

	class CHAOS_API File
	{
	public:
		File();
		File(const char* file);
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
	using FileList = std::vector<File>;

	typedef void(*PBCallback)(int);
	CHAOS_API void GetFileList(const std::string& folder, FileList& list, const std::string& types = "*", const PBCallback& update = nullptr);

	CHAOS_API void Copy(const File& from, const File& to, bool force = false);
	CHAOS_API void Move(const File& from, const File& to, bool force = false);
	CHAOS_API void Delete(const File& file);

	/// <summary>Split the string data by delimiter</summary>
	CHAOS_API std::vector<std::string> Split(const std::string& data, const std::string& delimiter);
	/// <summary>Remove all space and line break</summary>
	CHAOS_API std::string Shrink(const std::string& data);

	CHAOS_API std::string GetVersionInfo();
	/// <summary>ChaosMX Version Info</summary>
	CHAOS_API std::string GetMXVI(); 

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