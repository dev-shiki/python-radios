import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import orjson
import pytest
from aiodns import DNSResolver
from awesomeversion import AwesomeVersion

from radios.const import FilterBy, Order
from radios.exceptions import (
    RadioBrowserConnectionError,
    RadioBrowserConnectionTimeoutError,
    RadioBrowserError,
)
from radios.models import Country, Language, Station, Stats, Tag
from radios.radio_browser import RadioBrowser


@pytest.fixture
def mock_dns_resolver():
    """Mock the DNS resolver."""
    with patch("aiodns.DNSResolver", autospec=True) as mock_resolver_class:
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver
        
        # Create a mock result with a host attribute
        mock_result = MagicMock()
        mock_result.host = "api.radio-browser.info"
        mock_resolver.query.return_value = asyncio.Future()
        mock_resolver.query.return_value.set_result([mock_result])
        
        yield mock_resolver


@pytest.fixture
def mock_session():
    """Mock aiohttp ClientSession."""
    with patch("aiohttp.ClientSession", autospec=True) as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.text = AsyncMock()
        
        mock_session.request.return_value = mock_response
        
        yield mock_session, mock_response


@pytest.fixture
def radio_browser():
    """Create a RadioBrowser instance."""
    return RadioBrowser(user_agent="test_user_agent")


class TestRadioBrowser:
    """Tests for the RadioBrowser class."""

    @pytest.mark.asyncio
    async def test_stats(self, radio_browser, mock_dns_resolver, mock_session):
        """Test retrieving stats from the Radio Browser API."""
        _, mock_response = mock_session
        
        # Prepare mock response data
        stats_data = {
            "supported_version": 1,
            "software_version": "1.0.0",
            "status": "OK",
            "stations": 25000,
            "stations_broken": 125,
            "tags": 1500,
            "clicks_last_hour": 750,
            "clicks_last_day": 12500,
            "languages": 75,
            "countries": 150,
        }
        mock_response.text.return_value = orjson.dumps(stats_data)
        
        # Call the method
        result = await radio_browser.stats()
        
        # Verify the result
        assert isinstance(result, Stats)
        assert result.supported_version == 1
        assert result.software_version == AwesomeVersion("1.0.0")
        assert result.status == "OK"
        assert result.stations == 25000
        assert result.stations_broken == 125
        assert result.tags == 1500
        assert result.clicks_last_hour == 750
        assert result.clicks_last_day == 12500
        assert result.languages == 75
        assert result.countries == 150

    @pytest.mark.asyncio
    async def test_station_click(self, radio_browser, mock_dns_resolver, mock_session):
        """Test registering a click on a station."""
        mock_session_obj, mock_response = mock_session
        
        # Prepare mock response
        mock_response.text.return_value = "{}"
        
        # Call the method
        await radio_browser.station_click(uuid="test-uuid")
        
        # Verify the request was made correctly
        mock_session_obj.request.assert_called_once()
        call_args = mock_session_obj.request.call_args[0]
        assert call_args[0] == "GET"
        assert "url/test-uuid" in str(call_args[1])

    @pytest.mark.asyncio
    async def test_countries(self, radio_browser, mock_dns_resolver, mock_session):
        """Test retrieving countries from the Radio Browser API."""
        _, mock_response = mock_session
        
        # Prepare mock response data
        countries_data = [
            {"name": "US", "stationcount": "5000"},
            {"name": "XK", "stationcount": "100"},
            {"name": "DE", "stationcount": "3000"},
        ]
        mock_response.text.return_value = orjson.dumps(countries_data)
        
        # Call the method
        result = await radio_browser.countries()
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(country, Country) for country in result)
        
        # Check specific country data
        us_country = next(c for c in result if c.code == "US")
        assert us_country.name == "United States"
        assert us_country.station_count == "5000"
        
        kosovo_country = next(c for c in result if c.code == "XK")
        assert kosovo_country.name == "Kosovo"
        assert kosovo_country.station_count == "100"

    @pytest.mark.asyncio
    async def test_countries_with_parameters(self, radio_browser, mock_dns_resolver, mock_session):
        """Test retrieving countries with custom parameters."""
        mock_session_obj, mock_response = mock_session
        
        # Prepare mock response data
        countries_data = [
            {"name": "US", "stationcount": "5000"},
        ]
        mock_response.text.return_value = orjson.dumps(countries_data)
        
        # Call the method with parameters
        result = await radio_browser.countries(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True,
        )
        
        # Verify the request was made with correct parameters
        mock_session_obj.request.assert_called_once()
        _, kwargs = mock_session_obj.request.call_args
        assert kwargs["params"]["hidebroken"] == "true"
        assert kwargs["params"]["limit"] == 10
        assert kwargs["params"]["offset"] == 5
        assert kwargs["params"]["order"] == "stationcount"
        assert kwargs["params"]["reverse"] == "true"

    @pytest.mark.asyncio
    async def test_languages(self, radio_browser, mock_dns_resolver, mock_session):
        """Test retrieving languages from the Radio Browser API."""
        _, mock_response = mock_session
        
        # Prepare mock response data
        languages_data = [
            {"name": "english", "stationcount": "10000", "iso_639": "en"},
            {"name": "german", "stationcount": "5000", "iso_639": "de"},
            {"name": "spanish", "stationcount": "7000", "iso_639": "es"},
        ]
        mock_response.text.return_value = orjson.dumps(languages_data)
        
        # Call the method
        result = await radio_browser.languages()
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(language, Language) for language in result)
        
        # Check specific language data
        english = next(l for l in result if l.code == "en")
        assert english.name == "English"  # Should be title-cased
        assert english.station_count == "10000"
        
        german = next(l for l in result if l.code == "de")
        assert german.name == "German"
        assert german.station_count == "5000"

    @pytest.mark.asyncio
    async def test_languages_with_parameters(self, radio_browser, mock_dns_resolver, mock_session):
        """Test retrieving languages with custom parameters."""
        mock_session_obj, mock_response = mock_session
        
        # Prepare mock response data
        languages_data = [
            {"name": "english", "stationcount": "10000", "iso_639": "en"},
        ]
        mock_response.text.return_value = orjson.dumps(languages_data)
        
        # Call the method with parameters
        result = await radio_browser.languages(
            hide_broken=True,
            limit=20,
            offset=10,
            order=Order.STATION_COUNT,
            reverse=True,
        )
        
        # Verify the request was made with correct parameters
        mock_session_obj.request.assert_called_once()
        _, kwargs = mock_session_obj.request.call_args
        assert kwargs["params"]["hidebroken"] == "true"
        assert kwargs["params"]["limit"] == 20
        assert kwargs["params"]["offset"] == 10
        assert kwargs["params"]["order"] == "stationcount"
        assert kwargs["params"]["reverse"] == "true"

    @pytest.mark.asyncio
    async def test_search(self, radio_browser, mock_dns_resolver, mock_session):
        """Test searching for stations."""
        _, mock_response = mock_session
        
        # Prepare mock response data
        stations_data = [
            {
                "changeuuid": "change-uuid-1",
                "stationuuid": "station-uuid-1",
                "name": "Test Station 1",
                "url": "http://example.com/stream1",
                "url_resolved": "http://example.com/stream1",
                "homepage": "http://example.com",
                "favicon": "http://example.com/favicon.ico",
                "tags": "tag1,tag2",
                "country": "United States",
                "countrycode": "US",
                "state": "California",
                "language": "english",
                "languagecodes": "en",
                "votes": 10,
                "lastchangetime_iso8601": "2023-01-01T12:00:00Z",
                "codec": "MP3",
                "bitrate": 128,
                "hls": False,
                "lastcheckok": True,
                "lastchecktime_iso8601": "2023-01-01T12:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T12:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T12:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T12:00:00Z",
                "clickcount": 100,
                "clicktrend": 5,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "iso_3166_2": "US-CA"
            }
        ]
        mock_response.text.return_value = orjson.dumps(stations_data)
        
        # Call the method
        result = await radio_browser.search(name="Test")
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(station, Station) for station in result)
        
        # Check specific station data
        station = result[0]
        assert station.name == "Test Station 1"
        assert station.uuid == "station-uuid-1"
        assert station.url == "http://example.com/stream1"
        assert station.bitrate == 128
        assert station.country_code == "US"
        assert station.country == "United States"

    @pytest.mark.asyncio
    async def test_search_with_filter(self, radio_browser, mock_dns_resolver, mock_session):
        """Test searching for stations with filter parameters."""
        mock_session_obj, mock_response = mock_session
        
        # Prepare mock response data
        stations_data = [
            {
                "changeuuid": "change-uuid-1",
                "stationuuid": "station-uuid-1",
                "name": "Test Station 1",
                "url": "http://example.com/stream1",
                "url_resolved": "http://example.com/stream1",
                "homepage": "http://example.com",
                "favicon": "http://example.com/favicon.ico",
                "tags": "tag1,tag2",
                "country": "United States",
                "countrycode": "US",
                "state": "California",
                "language": "english",
                "languagecodes": "en",
                "votes": 10,
                "lastchangetime_iso8601": "2023-01-01T12:00:00Z",
                "codec": "MP3",
                "bitrate": 128,
                "hls": False,
                "lastcheckok": True,
                "lastchecktime_iso8601": "2023-01-01T12:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T12:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T12:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T12:00:00Z",
                "clickcount": 100,
                "clicktrend": 5,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "iso_3166_2": "US-CA"
            }
        ]
        mock_response.text.return_value = orjson.dumps(stations_data)
        
        # Call the method with filter parameters
        result = await radio_browser.search(
            filter_by=FilterBy.COUNTRY,
            filter_term="US",
            hide_broken=True,
            limit=10,
            offset=0,
            order=Order.NAME,
            reverse=False,
            name="Test",
            name_exact=True,
            country="US",
            country_exact=True,
            state_exact=True,
            language_exact=True,
            tag_exact=True,
            bitrate_min=128,
            bitrate_max=320,
        )
        
        # Verify the request was made with correct parameters
        mock_session_obj.request.assert_called_once()
        args, kwargs = mock_session_obj.request.call_args
        assert "stations/search/bycountry/US" in str(args[1])
        assert kwargs["params"]["hidebroken"] == "true"
        assert kwargs["params"]["limit"] == 10
        assert kwargs["params"]["offset"] == 0
        assert kwargs["params"]["order"] == "name"
        assert kwargs["params"]["reverse"] == "false"
        assert kwargs["params"]["name"] == "Test"
        assert kwargs["params"]["name_exact"] == "true"
        assert kwargs["params"]["country"] == "US"
        assert kwargs["params"]["country_exact"] == "true"
        assert kwargs["params"]["state_exact"] == "true"
        assert kwargs["params"]["language_exact"] == "true"
        assert kwargs["params"]["tag_exact"] == "true"
        assert kwargs["params"]["bitrate_min"] == 128
        assert kwargs["params"]["bitrate_max"] == 320

    @pytest.mark.asyncio
    async def test_station(self, radio_browser, mock_dns_resolver, mock_session):
        """Test retrieving a single station by UUID."""
        _, mock_response = mock_session
        
        # Prepare mock response data
        stations_data = [
            {
                "changeuuid": "change-uuid-1",
                "stationuuid": "test-uuid",
                "name": "Test Station 1",
                "url": "http://example.com/stream1",
                "url_resolved": "http://example.com/stream1",
                "homepage": "http://example.com",
                "favicon": "http://example.com/favicon.ico",
                "tags": "tag1,tag2",
                "country": "United States",
                "countrycode": "US",
                "state": "California",
                "language": "english",
                "languagecodes": "en",
                "votes": 10,
                "lastchangetime_iso8601": "2023-01-01T12:00:00Z",
                "codec": "MP3",
                "bitrate": 128,
                "hls": False,
                "lastcheckok": True,
                "lastchecktime_iso8601": "2023-01-01T12:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T12:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T12:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T12:00:00Z",
                "clickcount": 100,
                "clicktrend": 5,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "iso_3166_2": "US-CA"
            }
        ]
        mock_response.text.return_value = orjson.dumps(stations_data)
        
        # Call the method
        result = await radio_browser.station(uuid="test-uuid")
        
        # Verify the result
        assert isinstance(result, Station)
        assert result.uuid == "test-uuid"
        assert result.name == "Test Station 1"

    @pytest.mark.asyncio
    async def test_station_not_found(self, radio_browser, mock_dns_resolver, mock_session):
        """Test retrieving a station that doesn't exist."""
        _, mock_response = mock_session
        
        # Prepare empty response
        mock_response.text.return_value = "[]"
        
        # Call the method
        result = await radio_browser.station(uuid="nonexistent-uuid")
        
        # Verify the result is None
        assert result is None

    @pytest.mark.asyncio
    async def test_stations(self, radio_browser, mock_dns_resolver, mock_session):
        """Test retrieving stations."""
        _, mock_response = mock_session
        
        # Prepare mock response data
        stations_data = [
            {
                "changeuuid": "change-uuid-1",
                "stationuuid": "station-uuid-1",
                "name": "Test Station 1",
                "url": "http://example.com/stream1",
                "url_resolved": "http://example.com/stream1",
                "homepage": "http://example.com",
                "favicon": "http://example.com/favicon.ico",
                "tags": "tag1,tag2",
                "country": "United States",
                "countrycode": "US",
                "state": "California",
                "language": "english",
                "languagecodes": "en",
                "votes": 10,
                "lastchangetime_iso8601": "2023-01-01T12:00:00Z",
                "codec": "MP3",
                "bitrate": 128,
                "hls": False,
                "lastcheckok": True,
                "lastchecktime_iso8601": "2023-01-01T12:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T12:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T12:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T12:00:00Z",
                "clickcount": 100,
                "clicktrend": 5,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "iso_3166_2": "US-CA"
            },
            {
                "changeuuid": "change-uuid-2",
                "stationuuid": "station-uuid-2",
                "name": "Test Station 2",
                "url": "http://example.com/stream2",
                "url_resolved": "http://example.com/stream2",
                "homepage": "http://example.com",
                "favicon": "http://example.com/favicon.ico",
                "tags": "tag2,tag3",
                "country": "Germany",
                "countrycode": "DE",
                "state": "Berlin",
                "language": "german",
                "languagecodes": "de",
                "votes": 5,
                "lastchangetime_iso8601": "2023-01-02T12:00:00Z",
                "codec": "AAC",
                "bitrate": 256,
                "hls": True,
                "lastcheckok": True,
                "lastchecktime_iso8601": "2023-01-02T12:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-02T12:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-02T12:00:00Z",
                "clicktimestamp_iso8601": "2023-01-02T12:00:00Z",
                "clickcount": 50,
                "clicktrend": 2,
                "ssl_error": 0,
                "geo_lat": 52.5200,
                "geo_long": 13.4050,
                "has_extended_info": True,
                "iso_3166_2": "DE-BE"
            }
        ]
        mock_response.text.return_value = orjson.dumps(stations_data)
        
        # Call the method
        result = await radio_browser.stations()
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(station, Station) for station in result)
        
        # Check specific station data
        station1 = next(s for s in result if s.uuid == "station-uuid-1")
        assert station1.name == "Test Station 1"
        assert station1.country_code == "US"
        
        station2 = next(s for s in result if s.uuid == "station-uuid-2")
        assert station2.name == "Test Station 2"
        assert station2.country_code == "DE"

    @pytest.mark.asyncio
    async def test_stations_with_filter(self, radio_browser, mock_dns_resolver, mock_session):
        """Test retrieving stations with filter parameters."""
        mock_session_obj, mock_response = mock_session
        
        # Prepare mock response data
        stations_data = [
            {
                "changeuuid": "change-uuid-1",
                "stationuuid": "station-uuid-1",
                "name": "Test Station 1",
                "url": "http://example.com/stream1",
                "url_resolved": "http://example.com/stream1",
                "homepage": "http://example.com",
                "favicon": "http://example.com/favicon.ico",
                "tags": "tag1,tag2",
                "country": "United States",
                "countrycode": "US",
                "state": "California",
                "language": "english",
                "languagecodes": "en",
                "votes": 10,
                "lastchangetime_iso8601": "2023-01-01T12:00:00Z",
                "codec": "MP3",
                "bitrate": 128,
                "hls": False,
                "lastcheckok": True,
                "lastchecktime_iso8601": "2023-01-01T12:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T12:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T12:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T12:00:00Z",
                "clickcount": 100,
                "clicktrend": 5,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "iso_3166_2": "US-CA"
            }
        ]
        mock_response.text.return_value = orjson.dumps(stations_data)
        
        # Call the method with filter parameters
        result = await radio_browser.stations(
            filter_by=FilterBy.COUNTRY,
            filter_term="US",
            hide_broken=True,
            limit=10,
            offset=0,
            order=Order.NAME,
            reverse=False,
        )
        
        # Verify the request was made with correct parameters
        mock_session_obj.request.assert_called_once()
        args, kwargs = mock_session_obj.request.call_args
        assert "stations/bycountry/US" in str(args[1])
        assert kwargs["params"]["hidebroken"] == "true"
        assert kwargs["params"]["limit"] == 10
        assert kwargs["params"]["offset"] == 0
        assert kwargs["params"]["order"] == "name"
        assert kwargs["params"]["reverse"] == "false"

    @pytest.mark.asyncio
    async def test_tags(self, radio_browser, mock_dns_resolver, mock_session):
        """Test retrieving tags."""
        _, mock_response = mock_session
        
        # Prepare mock response data
        tags_data = [
            {"name": "rock", "stationcount": "5000"},
            {"name": "pop", "stationcount": "4000"},
            {"name": "jazz", "stationcount": "2000"},
        ]
        mock_response.text.return_value = orjson.dumps(tags_data)
        
        # Call the method
        result = await radio_browser.tags()
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(tag, Tag) for tag in result)
        
        # Check specific tag data
        rock_tag = next(t for t in result if t.name == "rock")
        assert rock_tag.station_count == "5000"
        
        pop_tag = next(t for t in result if t.name == "pop")
        assert pop_tag.station_count == "4000"

    @pytest.mark.asyncio
    async def test_tags_with_parameters(self, radio_browser, mock_dns_resolver, mock_session):
        """Test retrieving tags with custom parameters."""
        mock_session_obj, mock_response = mock_session
        
        # Prepare mock response data
        tags_data = [
            {"name": "rock", "stationcount": "5000"},
        ]
        mock_response.text.return_value = orjson.dumps(tags_data)
        
        # Call the method with parameters
        result = await radio_browser.tags(
            hide_broken=True,
            limit=20,
            offset=10,
            order=Order.STATION_COUNT,
            reverse=True,
        )
        
        # Verify the request was made with correct parameters
        mock_session_obj.request.assert_called_once()
        _, kwargs = mock_session_obj.request.call_args
        assert kwargs["params"]["hidebroken"] == "true"
        assert kwargs["params"]["limit"] == 20
        assert kwargs["params"]["offset"] == 10
        assert kwargs["params"]["order"] == "stationcount"
        assert kwargs["params"]["reverse"] == "true"

    @pytest.mark.asyncio
    async def test_close(self, radio_browser):
        """Test closing the session."""
        # Create a mock session
        radio_browser.session = AsyncMock()
        radio_browser._close_session = True
        
        # Call the method
        await radio_browser.close()
        
        # Verify the session was closed
        radio_browser.session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_session(self, radio_browser):
        """Test closing when there's no session."""
        # Ensure there's no session
        radio_browser.session = None
        
        # Call the method (should not raise)
        await radio_browser.close()

    @pytest.mark.asyncio
    async def test_close_not_owned(self, radio_browser):
        """Test not closing a session we don't own."""
        # Create a mock session but mark it as not owned
        radio_browser.session = AsyncMock()
        radio_browser._close_session = False
        
        # Call the method
        await radio_browser.close()
        
        # Verify the session was not closed
        radio_browser.session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_aenter(self, radio_browser):
        """Test async context manager entry."""
        result = await radio_browser.__aenter__()
        assert result is radio_browser

    @pytest.mark.asyncio
    async def test_aexit(self, radio_browser):
        """Test async context manager exit."""
        # Create a mock close method
        radio_browser.close = AsyncMock()
        
        # Call the method
        await radio_browser.__aexit__(None, None, None)
        
        # Verify close was called
        radio_browser.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_connection_error(self, radio_browser, mock_dns_resolver):
        """Test handling connection errors."""
        # Create a session that raises a connection error
        session_mock = AsyncMock()
        session_mock.request.side_effect = aiohttp.ClientError("Connection error")
        
        # Set up the radio browser with the mock session
        radio_browser.session = session_mock
        radio_browser._host = "api.radio-browser.info"
        
        # Call a method that uses _request and expect an error
        with pytest.raises(RadioBrowserConnectionError):
            await radio_browser.stats()
        
        # Verify the host was reset
        assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_timeout_error(self, radio_browser, mock_dns_resolver):
        """Test handling timeout errors."""
        # Create a session that raises a timeout error
        session_mock = AsyncMock()
        session_mock.request.side_effect = asyncio.TimeoutError("Timeout")
        
        # Set up the radio browser with the mock session
        radio_browser.session = session_mock
        radio_browser._host = "api.radio-browser.info"
        
        # Call a method that uses _request and expect an error
        with pytest.raises(RadioBrowserConnectionTimeoutError):
            await radio_browser.stats()
        
        # Verify the host was reset
        assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_invalid_content_type(self, radio_browser, mock_dns_resolver):
        """Test handling invalid content type responses."""
        # Create a mock response with non-JSON content type
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text.return_value = "<html>Not JSON</html>"
        
        # Create a session that returns the mock response
        session_mock = AsyncMock()
        session_mock.request.return_value = mock_response
        
        # Set up the radio browser with the mock session
        radio_browser.session = session_mock
        radio_browser._host = "api.radio-browser.info"
        
        # Call a method that uses _request and expect an error
        with pytest.raises(RadioBrowserError):
            await radio_browser.stats()