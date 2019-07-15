#include "core/core.hpp"

#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <mutex>

namespace chaos
{
	DEFINE_STRING(log_dir, "", "", "to set log dir");
	DEFINE_INT(log_level, 0, "", "to set log level");

	static std::string log_name = "";
	static int log_level = 0;
	static std::mutex mtx;
	static std::map<LogSeverity, unsigned short> colors = { {INFO, 0x07}, {WARNING, 0x0E}, {ERROR, 0x04}, {FATAL, 0x04} };

	auto ToString = [](LogSeverity se)->std::string {
		switch (se)
		{
		case INFO:
			return "INFO";
		case WARNING:
			return "WARNING";
		case ERROR:
			return "ERROR";
		case FATAL:
		default:
			return "FATAL";
		}
	};

	LogMessage::LogMessage(const char* file, int line, LogSeverity severity)
		: severity(severity)
	{
		File _file(file);

		//// Get stamp
		time_t time_stamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		tm time;
		localtime_s(&time, &time_stamp);

		if (severity >= log_level)
		{
			message_data << "[" << ToString(severity) << " " << std::setfill('0')
				<< std::setw(4) << time.tm_year + 1900 << "-"
				<< std::setw(2) << time.tm_mon + 1 << "-"
				<< std::setw(2) << time.tm_mday << " "
				<< std::setw(2) << time.tm_hour << ":"
				<< std::setw(2) << time.tm_min << ":"
				<< std::setw(2) << time.tm_sec << " "
				<< _file.Name << "." << _file.Type << ":" << line << "] ";
			// Head likes "[INFO 2018-05-21 17:31:04 xx.cpp:21]"
		}
	}

	LogMessage::~LogMessage()
	{
		Flush();

		// if FATAL, then abort
		if (severity == FATAL) abort();
	}

	std::stringstream& chaos::LogMessage::Stream()
	{
		return message_data;
	}

	void LogMessage::Flush()
	{
		mtx.lock();
		if (severity >= log_level)
		{
			std::string message = message_data.str();

			SetConsoleTextColor(colors[severity]);
			std::cout << message << std::endl;
			SetConsoleTextColor(0x07);

			if (!flag_log_dir.empty())
			{
				// Save Log Message
				std::fstream log_file(flag_log_dir + "\\" + log_name, std::ios::out | std::ios::app);
				if (log_file.is_open())
				{
					log_file << message << std::endl;
					log_file.close();
				}
			}
		}
		mtx.unlock();
	}


	void InitLogging(const std::string argv0, int level)
	{
		time_t time_stamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		tm time;
		localtime_s(&time, &time_stamp);

		std::stringstream ss;
		ss << File(argv0).Name << "." << std::setfill('0')
			<< std::setw(4) << time.tm_year + 1900 << "."
			<< std::setw(2) << time.tm_mon + 1 << "."
			<< std::setw(2) << time.tm_mday << "."
			<< std::setw(2) << time.tm_hour << "."
			<< std::setw(2) << time.tm_min << "."
			<< std::setw(2) << time.tm_sec << ".LOG";
		// File name likes "exe_name.2018.01.15.21.35.03.LOG"

		log_name = ss.str();
		log_level = std::max(level, flag_log_level);
	}
}