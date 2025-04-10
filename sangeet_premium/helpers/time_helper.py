# ==============================================================================
# time_helper.py
# ==============================================================================
# Description:
#   This module provides utility classes and functions for handling time-related
#   operations, specifically focusing on conversions between UTC and IST
#   (Indian Standard Time), time formatting, and potentially time synchronization.
#
# Classes:
#   - TimeConverter: Handles manual UTC to IST conversion and formatting.
#   - TimeSync: Uses the `pytz` library for more robust timezone handling,
#               including IST, and provides formatting options.
# ==============================================================================

from datetime import datetime, timedelta, timezone
import logging
# import ntplib # This import was present but unused in the original code
import pytz

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --- TimeConverter Class ---
class TimeConverter:
    """
    Handles time conversion between UTC and Indian Standard Time (IST) manually.

    Note: This class performs manual offset calculation. For more robust timezone
    handling, especially considering potential DST changes (though not applicable
    to IST), using a library like `pytz` (as in `TimeSync`) is generally recommended.
    """

    # IST offset from UTC is +5:30
    IST_OFFSET_HOURS = 5
    IST_OFFSET_MINUTES = 30

    @classmethod
    def utc_to_ist(cls, utc_dt):
        """
        Convert a naive UTC datetime object to an IST datetime object using manual offset.

        Args:
            utc_dt (datetime): A datetime object assumed to be in UTC (naive).

        Returns:
            datetime: The corresponding datetime object in IST (naive), or None if input is None.
                      Returns the original object on error.
        """
        if not utc_dt:
            return None
        if not isinstance(utc_dt, datetime):
             logger.error(f"Input must be a datetime object, got {type(utc_dt)}")
             return utc_dt # Return original on type error

        # Ensure input is treated as naive UTC before applying offset
        utc_dt = utc_dt.replace(tzinfo=None)

        try:
            # Create the timedelta for the IST offset
            ist_offset = timedelta(hours=cls.IST_OFFSET_HOURS, minutes=cls.IST_OFFSET_MINUTES)

            # Add the offset to the UTC time
            ist_dt = utc_dt + ist_offset
            return ist_dt

        except Exception as e:
            logger.error(f"UTC to IST conversion error: {e}")
            return utc_dt # Return original datetime on error

    @classmethod
    def format_ist_timestamp(cls, dt, include_timezone=True):
        """
        Format a datetime object into a readable IST timestamp string.

        Assumes the input datetime `dt` should be treated as UTC before conversion.

        Args:
            dt (datetime): The datetime object (assumed UTC) to format.
            include_timezone (bool): Whether to append " IST" to the formatted string.

        Returns:
            str: The formatted timestamp string (e.g., "YYYY-MM-DD HH:MM:SS AM/PM IST")
                 or "Invalid Date" on error or if input is None.
        """
        if not dt:
            return "Invalid Date"
        if not isinstance(dt, datetime):
            logger.error(f"Input must be a datetime object, got {type(dt)}")
            return "Invalid Date"

        try:
            # Convert the input datetime (assumed UTC) to IST
            ist_dt = cls.utc_to_ist(dt)
            if ist_dt is None: # Check if conversion itself failed
                 return "Invalid Date"

            # Format the IST datetime
            formatted = ist_dt.strftime('%Y-%m-%d %I:%M:%S %p') # 12-hour format with AM/PM
            if include_timezone:
                formatted += " IST"

            return formatted

        except Exception as e:
            logger.error(f"IST formatting error: {e}")
            return "Invalid Date"

    @classmethod
    def format_relative_time(cls, dt):
        """
        Format a datetime object as a relative time string (e.g., '2 hours ago').

        Compares the input datetime (assumed UTC) to the current *local* system time
        after converting the input datetime to IST.

        Args:
            dt (datetime): The datetime object (assumed UTC) to format relatively.

        Returns:
            str: A human-readable relative time string (e.g., "just now", "5 minutes ago",
                 "3 hours ago", "2 days ago") or the absolute date ("DD Mon YYYY")
                 if older than a week. Returns "Unknown time" on error or if input is None.
        """
        if not dt:
            return "Unknown time"
        if not isinstance(dt, datetime):
            logger.error(f"Input must be a datetime object, got {type(dt)}")
            return "Unknown time"

        try:
            # Get the current time (naive, local timezone of the system)
            # Consider using timezone-aware comparison if precision across DST is critical
            now = datetime.now()

            # Convert the input datetime (assumed UTC) to IST
            ist_dt = cls.utc_to_ist(dt)
            if ist_dt is None: # Check if conversion itself failed
                return "Unknown time"


            # Calculate the difference between now (local) and the IST time
            # Note: This compares naive local time with naive IST time.
            # This is generally okay if the server runs in IST or for relative display,
            # but be mindful of potential inaccuracies if server TZ differs significantly
            # or if DST transitions are involved for the local time.
            diff = now - ist_dt

            seconds = diff.total_seconds()

            # Handle future dates (though unlikely if dt is from the past)
            if seconds < 0:
                return "in the future" # Or format differently

            if seconds < 60:
                return "just now"
            elif seconds < 3600: # Less than 1 hour
                minutes = int(seconds / 60)
                return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
            elif seconds < 86400: # Less than 1 day (24 * 3600)
                hours = int(seconds / 3600)
                return f"{hours} hour{'s' if hours > 1 else ''} ago"
            elif seconds < 604800: # Less than 1 week (7 * 86400)
                days = int(seconds / 86400)
                return f"{days} day{'s' if days > 1 else ''} ago"
            else: # Older than a week
                # Format as absolute date
                return ist_dt.strftime('%d %b %Y')

        except Exception as e:
            logger.error(f"Relative time formatting error: {e}")
            return "Unknown time"


# --- TimeSync Class ---
class TimeSync:
    """
    Handles time operations using the `pytz` library for accurate timezone awareness.

    Provides methods to get the current time in IST, parse datetime strings,
    and format datetimes, including relative formatting. This approach is generally
    more reliable than manual offset calculations.
    """
    def __init__(self):
        """Initializes the TimeSync class by setting the IST timezone."""
        try:
            self.ist = pytz.timezone('Asia/Kolkata')
        except pytz.UnknownTimeZoneError:
            logger.error("Could not find timezone 'Asia/Kolkata'. Make sure pytz is installed correctly.")
            # Fallback or raise error - using UTC as a safe default if IST fails
            self.ist = pytz.utc

    def get_current_time(self):
        """
        Get the current time, localized to the IST timezone.

        Returns:
            datetime: A timezone-aware datetime object representing the current time in IST.
        """
        # Get current UTC time and convert it to IST
        return datetime.now(pytz.utc).astimezone(self.ist)

    def parse_datetime(self, dt_str):
        """
        Parse an ISO format datetime string and convert it to IST.

        Assumes the input string is in a format recognizable by `datetime.fromisoformat`.
        If the string has timezone info, it's used; otherwise, it's assumed naive
        and then localized to IST (behavior might depend on exact ISO format).
        For consistent results, ensure input strings have timezone info (like 'Z' or '+HH:MM').

        Args:
            dt_str (str): The datetime string in ISO format.

        Returns:
            datetime: A timezone-aware datetime object in IST. Returns None on parsing error.
        """
        try:
            # Attempt to parse the ISO string
            dt_naive_or_aware = datetime.fromisoformat(dt_str)

            # If the parsed datetime is naive (no timezone info)
            if dt_naive_or_aware.tzinfo is None or dt_naive_or_aware.tzinfo.utcoffset(dt_naive_or_aware) is None:
                # Localize naive time to IST. This assumes the naive time *represents* IST.
                # If the naive time represents UTC, you should first make it UTC aware:
                # dt_aware = pytz.utc.localize(dt_naive_or_aware)
                # return dt_aware.astimezone(self.ist)
                # Assuming naive dt_str represents IST time directly:
                 return self.ist.localize(dt_naive_or_aware)
            else:
                # If it's already timezone-aware, convert it to IST
                return dt_naive_or_aware.astimezone(self.ist)
        except ValueError as e:
            logger.error(f"Error parsing datetime string '{dt_str}': {e}")
            return None
        except Exception as e: # Catch other potential errors
             logger.error(f"Unexpected error parsing datetime string '{dt_str}': {e}")
             return None


    def format_time(self, dt, relative=False):
        """
        Formats a datetime object (or string) into a specified string format in IST.

        Handles both timezone-aware and naive datetime objects. Naive objects are
        assumed to represent IST. Also accepts ISO datetime strings.

        Args:
            dt (datetime | str): The datetime object or ISO string to format.
            relative (bool): If True, format as relative time (e.g., "5m ago").
                             If False, format as 'YYYY-MM-DD HH:MM:SS'.

        Returns:
            str: The formatted time string, or an error message if input is invalid.
        """
        if isinstance(dt, str):
            dt = self.parse_datetime(dt) # Attempt to parse if it's a string

        if not isinstance(dt, datetime):
            logger.warning(f"Invalid input type for format_time: {type(dt)}. Expected datetime or str.")
            return "Invalid Date/Time Input"

        # Ensure the datetime is IST timezone-aware for correct formatting/comparison
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
             # Assume naive datetime represents IST
             dt_aware = self.ist.localize(dt)
        elif dt.tzinfo != self.ist :
             dt_aware = dt.astimezone(self.ist)
        else:
             dt_aware = dt # Already in IST and aware


        if relative:
            try:
                now = self.get_current_time() # Get current time in IST
                diff = now - dt_aware

                seconds = diff.total_seconds()

                if seconds < 0:
                     return "in the future" # Or format appropriately
                if seconds < 60:
                    return f"{int(seconds)}s ago" # Added seconds for more granularity
                minutes = seconds / 60
                if minutes < 60:
                    return f"{int(minutes)}m ago"
                hours = minutes / 60
                if hours < 24:
                    return f"{int(hours)}h ago"
                days = hours / 24
                # Simple day-based relative formatting
                return f"{int(days)}d ago"
                # Could add weeks/months/years if needed
            except Exception as e:
                logger.error(f"Error calculating relative time: {e}")
                # Fallback to absolute time on relative calculation error
                return dt_aware.strftime('%Y-%m-%d %H:%M:%S')
        else:
            # Format as absolute time in IST (24-hour format)
            return dt_aware.strftime('%Y-%m-%d %H:%M:%S')

# Example Usage (optional)
# if __name__ == "__main__":
#     # --- TimeConverter Example ---
#     print("--- TimeConverter Examples ---")
#     utc_now = datetime.utcnow()
#     print(f"UTC Now (naive): {utc_now}")
#     ist_now_manual = TimeConverter.utc_to_ist(utc_now)
#     print(f"IST Now (manual): {ist_now_manual}")
#     print(f"Formatted IST (manual): {TimeConverter.format_ist_timestamp(utc_now)}")
#     past_time_utc = datetime.utcnow() - timedelta(hours=3, minutes=15)
#     print(f"Relative Time (manual): {TimeConverter.format_relative_time(past_time_utc)}")
#     way_past_time_utc = datetime.utcnow() - timedelta(days=10)
#     print(f"Relative Time (> week, manual): {TimeConverter.format_relative_time(way_past_time_utc)}")
#     print("-" * 20)

#     # --- TimeSync Example ---
#     print("--- TimeSync Examples ---")
#     tsync = TimeSync()
#     current_ist_pytz = tsync.get_current_time()
#     print(f"Current IST (pytz): {current_ist_pytz}")
#     print(f"Formatted IST (pytz, absolute): {tsync.format_time(current_ist_pytz)}")
#     past_time_pytz = tsync.get_current_time() - timedelta(minutes=45)
#     print(f"Formatted IST (pytz, relative <1h): {tsync.format_time(past_time_pytz, relative=True)}")
#     past_time_pytz_hours = tsync.get_current_time() - timedelta(hours=5)
#     print(f"Formatted IST (pytz, relative <1d): {tsync.format_time(past_time_pytz_hours, relative=True)}")
#     past_time_pytz_days = tsync.get_current_time() - timedelta(days=3)
#     print(f"Formatted IST (pytz, relative >1d): {tsync.format_time(past_time_pytz_days, relative=True)}")

#     # Example parsing
#     iso_str = "2023-10-26T10:00:00Z" # UTC time
#     parsed_dt = tsync.parse_datetime(iso_str)
#     if parsed_dt:
#         print(f"Parsed '{iso_str}' to IST: {parsed_dt}")
#         print(f"Formatted Parsed (absolute): {tsync.format_time(parsed_dt)}")
#     else:
#         print(f"Failed to parse '{iso_str}'")

#     iso_str_naive = "2023-10-27T14:30:00" # Assumed IST if parsed by TimeSync
#     parsed_dt_naive = tsync.parse_datetime(iso_str_naive)
#     if parsed_dt_naive:
#          print(f"Parsed '{iso_str_naive}' (naive assumed IST): {parsed_dt_naive}")
#          print(f"Formatted Parsed Naive (absolute): {tsync.format_time(parsed_dt_naive)}")
#     else:
#         print(f"Failed to parse '{iso_str_naive}'")
#     print("-" * 20)