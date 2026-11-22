Template : [Blocking] what - impact -fix

Top 3 problems :

1. In TripRequest, check_dates, end and start comparator is <= - impact is if the user ask for a single day it will end in error - fix : < instead of <=
2. In check_forecast_window, the timedelta is off by one - the impact is that the config of the horizon date is not respected, and goes further to 1 (here 17 days instead of 16) - fix : timedelta(delta=horizon_days - 1) 
3. Cache with no TTL in weather.py for get_forecast - impact is during the whole process, the data will never be refreshed as long as the script is running - fix : implement a timestamp for each cache entry and set a TTL to a value that match the geo API refresh rate, let's say 1 hour

List of points : either Blocking, Should Fix, Nit

Blocking:

- Cache with no TTL in weather.py for get_forecast - impact is during the whole process, the data will never be refreshed as long as the script is running - fix : implement a timestamp for each cache entry and set a TTL to a value that match the geo API refresh rate, let's say 1 hour
- In check_forecast_window, the timedelta is off by one - the impact is that the config of the horizon date is not respected, and goes further to 1 (here 17 days instead of 16) - fix : timedelta(delta=horizon_days - 1) 
- In TripRequest, check_dates, end and start comparator is <= - impact is if the user ask for a single day it will end in error - fix : < instead of <=


Should Fix:

- Logging mistral API key in chat line 41
- answer() is a monolith function, a better design would be to split into several functions, better testability.
- In chat.py line 89, there is a blind Exception that catches everything (including kill signals) - impact is difficulty to read the error + blocking killswitches - fix : add exc and raise TripErrors or another Error class custom made 
- In get_forecast, logging information is printed instead of logging line 61

Nits:

- Logger httpx as warning, keep it as info to avoid overwhelming unnecessary logs
- Unused argument max_results in the geocode function. The parameter is already including a 10 element limit so no underlying issue here, but the argument should be removed, or implemented if needed.
