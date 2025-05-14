import asyncio
import socket
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
        mock_resolver = Mock(spec=DNSResolver)
        mock_resolver_class.return_value = mock_resolver
        
        # Create a mock result with a host attribute
        mock_result = MagicMock()
        mock_result.host = "api.radio-browser.info"
        
        # Set up the query method to return a list with the mock result
        mock_resolver.query = AsyncMock(return_value=[mock_result])
        
        yield mock_resolver


@pytest.fixture
def mock_session():
    """Mock aiohttp ClientSession."""
    with patch("aiohttp.ClientSession", autospec=True) as mock_session_class:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session_class.return_value = mock_session
        
        # Mock the response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.text = AsyncMock()
        
        # Mock the request method
        mock_session.request = AsyncMock(return_value=mock_response)
        
        yield mock_session, mock_response


@pytest.fixture
def radio_browser(mock_session):
    """Create a RadioBrowser instance with a mocked session."""
    browser = RadioBrowser(user_agent="test_agent")
    browser.session = mock_session[0]
    browser._host = "api.radio-browser.info"
    browser._close_session = False
    return browser


class TestRadioBrowser:
    """Tests for the RadioBrowser class."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test initialization of RadioBrowser."""
        browser = RadioBrowser(user_agent="test_agent")
        assert browser.user_agent == "test_agent"
        assert browser.request_timeout == 8.0
        assert browser.session is None
        assert browser._close_session is False
        assert browser._host is None

    @pytest.mark.asyncio
    async def test_request_dns_resolution(self, mock_dns_resolver):
        """Test _request method resolves DNS."""
        browser = RadioBrowser(user_agent="test_agent")
        
        # Mock session and response
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.text = AsyncMock(return_value='{"key": "value"}')
        mock_session.request = AsyncMock(return_value=mock_response)
        
        # Set the session
        browser.session = mock_session
        
        # Call _request
        result = await browser._request("test")
        
        # Verify DNS resolution was performed
        mock_dns_resolver.query.assert_awaited_once_with(
            "_api._tcp.radio-browser.info", "SRV"
        )
        
        # Verify the host was set
        assert browser._host == "api.radio-browser.info"
        
        # Verify the request was made
        mock_session.request.assert_awaited_once()
        assert result == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_request_with_params(self, mock_session):
        """Test _request method with parameters."""
        session, response = mock_session
        response.text.return_value = '{"key": "value"}'
        
        browser = RadioBrowser(user_agent="test_agent")
        browser.session = session
        browser._host = "api.radio-browser.info"
        
        params = {"param1": "value1", "param2": True}
        await browser._request("test", params=params)
        
        # Verify boolean params are converted to lowercase strings
        session.request.assert_awaited_once()
        call_args = session.request.call_args[1]
        assert call_args["params"]["param2"] == "true"

    @pytest.mark.asyncio
    async def test_request_timeout(self, mock_session):
        """Test _request method handles timeout."""
        session, _ = mock_session
        session.request.side_effect = asyncio.TimeoutError()
        
        browser = RadioBrowser(user_agent="test_agent")
        browser.session = session
        browser._host = "api.radio-browser.info"
        
        with pytest.raises(RadioBrowserConnectionTimeoutError):
            await browser._request("test")
        
        # Verify host is reset on error
        assert browser._host is None

    @pytest.mark.asyncio
    async def test_request_connection_error(self, mock_session):
        """Test _request method handles connection error."""
        session, _ = mock_session
        session.request.side_effect = aiohttp.ClientError()
        
        browser = RadioBrowser(user_agent="test_agent")
        browser.session = session
        browser._host = "api.radio-browser.info"
        
        with pytest.raises(RadioBrowserConnectionError):
            await browser._request("test")
        
        # Verify host is reset on error
        assert browser._host is None

    @pytest.mark.asyncio
    async def test_request_socket_error(self, mock_session):
        """Test _request method handles socket error."""
        session, _ = mock_session
        session.request.side_effect = socket.gaierror()
        
        browser = RadioBrowser(user_agent="test_agent")
        browser.session = session
        browser._host = "api.radio-browser.info"
        
        with pytest.raises(RadioBrowserConnectionError):
            await browser._request("test")
        
        # Verify host is reset on error
        assert browser._host is None

    @pytest.mark.asyncio
    async def test_request_invalid_content_type(self, mock_session):
        """Test _request method handles invalid content type."""
        session, response = mock_session
        response.headers = {"Content-Type": "text/html"}
        response.text.return_value = "<html>Not JSON</html>"
        
        browser = RadioBrowser(user_agent="test_agent")
        browser.session = session
        browser._host = "api.radio-browser.info"
        
        with pytest.raises(RadioBrowserError):
            await browser._request("test")

    @pytest.mark.asyncio
    async def test_stats(self, radio_browser, mock_session):
        """Test stats method."""
        _, response = mock_session
        stats_data = {
            "supported_version": 1,
            "software_version": "2.0.0",
            "status": "OK",
            "stations": 1000,
            "stations_broken": 10,
            "tags": 500,
            "clicks_last_hour": 100,
            "clicks_last_day": 1000,
            "languages": 50,
            "countries": 100,
        }
        response.text.return_value = orjson.dumps(stats_data).decode()
        
        stats = await radio_browser.stats()
        
        assert isinstance(stats, Stats)
        assert stats.supported_version == 1
        assert stats.software_version == AwesomeVersion("2.0.0")
        assert stats.status == "OK"
        assert stats.stations == 1000
        assert stats.stations_broken == 10
        assert stats.tags == 500
        assert stats.clicks_last_hour == 100
        assert stats.clicks_last_day == 1000
        assert stats.languages == 50
        assert stats.countries == 100

    @pytest.mark.asyncio
    async def test_station_click(self, radio_browser, mock_session):
        """Test station_click method."""
        session, response = mock_session
        response.text.return_value = "{}"
        
        await radio_browser.station_click(uuid="test-uuid")
        
        session.request.assert_awaited_once()
        call_args = session.request.call_args
        assert "url/test-uuid" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_countries(self, radio_browser, mock_session):
        """Test countries method."""
        _, response = mock_session
        countries_data = [
            {"name": "US", "stationcount": "100"},
            {"name": "GB", "stationcount": "50"},
            {"name": "XK", "stationcount": "10"},  # Kosovo special case
        ]
        response.text.return_value = orjson.dumps(countries_data).decode()
        
        countries = await radio_browser.countries()
        
        assert len(countries) == 3
        assert isinstance(countries[0], Country)
        
        # Check Kosovo special case
        kosovo = next(c for c in countries if c.code == "XK")
        assert kosovo.name == "Kosovo"
        
        # Check normal country resolution
        us = next(c for c in countries if c.code == "US")
        assert us.name == "United States"
        
        # Check station count
        assert us.station_count == "100"

    @pytest.mark.asyncio
    async def test_countries_with_parameters(self, radio_browser, mock_session):
        """Test countries method with parameters."""
        session, response = mock_session
        response.text.return_value = orjson.dumps([
            {"name": "US", "stationcount": "100"}
        ]).decode()
        
        await radio_browser.countries(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True
        )
        
        session.request.assert_awaited_once()
        call_args = session.request.call_args[1]
        assert call_args["params"]["hidebroken"] == "true"
        assert call_args["params"]["limit"] == 10
        assert call_args["params"]["offset"] == 5
        assert call_args["params"]["order"] == "stationcount"
        assert call_args["params"]["reverse"] == "true"

    @pytest.mark.asyncio
    async def test_languages(self, radio_browser, mock_session):
        """Test languages method."""
        _, response = mock_session
        languages_data = [
            {"name": "english", "stationcount": "100", "iso_639": "en"},
            {"name": "spanish", "stationcount": "50", "iso_639": "es"},
            {"name": "unknown", "stationcount": "10", "iso_639": None},
        ]
        response.text.return_value = orjson.dumps(languages_data).decode()
        
        languages = await radio_browser.languages()
        
        assert len(languages) == 3
        assert isinstance(languages[0], Language)
        
        # Check name capitalization
        english = next(l for l in languages if l.code == "en")
        assert english.name == "English"
        
        # Check station count
        assert english.station_count == "100"
        
        # Check favicon property
        assert english.favicon == "https://flagcdn.com/256x192/en.png"
        
        # Check language with no code
        unknown = next(l for l in languages if l.code is None)
        assert unknown.favicon is None

    @pytest.mark.asyncio
    async def test_languages_with_parameters(self, radio_browser, mock_session):
        """Test languages method with parameters."""
        session, response = mock_session
        response.text.return_value = orjson.dumps([
            {"name": "english", "stationcount": "100", "iso_639": "en"}
        ]).decode()
        
        await radio_browser.languages(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True
        )
        
        session.request.assert_awaited_once()
        call_args = session.request.call_args[1]
        assert call_args["params"]["hidebroken"] == "true"
        assert call_args["params"]["limit"] == 10
        assert call_args["params"]["offset"] == 5
        assert call_args["params"]["order"] == "stationcount"
        assert call_args["params"]["reverse"] == "true"

    @pytest.mark.asyncio
    async def test_search(self, radio_browser, mock_session):
        """Test search method."""
        session, response = mock_session
        station_data = [{
            "changeuuid": "change-uuid",
            "stationuuid": "station-uuid",
            "name": "Test Station",
            "url": "http://example.com/stream",
            "url_resolved": "http://example.com/stream",
            "homepage": "http://example.com",
            "favicon": "http://example.com/favicon.ico",
            "tags": "tag1,tag2",
            "country": "United States",
            "countrycode": "US",
            "state": "California",
            "language": "English",
            "languagecodes": "en",
            "votes": 10,
            "clickcount": 100,
            "clicktrend": 5,
            "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
            "codec": "MP3",
            "bitrate": 128,
            "hls": False,
            "lastcheckok": True,
            "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
            "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
            "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
            "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
            "ssl_error": 0,
            "geo_lat": 37.7749,
            "geo_long": -122.4194,
            "has_extended_info": True,
            "iso_3166_2": "US-CA"
        }]
        response.text.return_value = orjson.dumps(station_data).decode()
        
        stations = await radio_browser.search(
            filter_by=FilterBy.NAME,
            filter_term="test",
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.BITRATE,
            reverse=True,
            name="Test",
            name_exact=True,
            country="US",
            country_exact=True,
            state_exact=True,
            language_exact=True,
            tag_exact=True,
            bitrate_min=64,
            bitrate_max=320
        )
        
        assert len(stations) == 1
        assert isinstance(stations[0], Station)
        assert stations[0].name == "Test Station"
        assert stations[0].uuid == "station-uuid"
        
        session.request.assert_awaited_once()
        call_args = session.request.call_args
        assert "stations/search/byname/test" in str(call_args[0][1])
        params = call_args[1]["params"]
        assert params["hidebroken"] == "true"
        assert params["limit"] == 10
        assert params["offset"] == 5
        assert params["order"] == "bitrate"
        assert params["reverse"] == "true"
        assert params["name"] == "Test"
        assert params["name_exact"] == "true"
        assert params["country"] == "US"
        assert params["country_exact"] == "true"
        assert params["state_exact"] == "true"
        assert params["language_exact"] == "true"
        assert params["tag_exact"] == "true"
        assert params["bitrate_min"] == 64
        assert params["bitrate_max"] == 320

    @pytest.mark.asyncio
    async def test_search_without_filter(self, radio_browser, mock_session):
        """Test search method without filter parameters."""
        session, response = mock_session
        station_data = [{
            "changeuuid": "change-uuid",
            "stationuuid": "station-uuid",
            "name": "Test Station",
            "url": "http://example.com/stream",
            "url_resolved": "http://example.com/stream",
            "homepage": "http://example.com",
            "favicon": "http://example.com/favicon.ico",
            "tags": "tag1,tag2",
            "country": "United States",
            "countrycode": "US",
            "state": "California",
            "language": "English",
            "languagecodes": "en",
            "votes": 10,
            "clickcount": 100,
            "clicktrend": 5,
            "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
            "codec": "MP3",
            "bitrate": 128,
            "hls": False,
            "lastcheckok": True,
            "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
            "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
            "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
            "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
            "ssl_error": 0,
            "geo_lat": 37.7749,
            "geo_long": -122.4194,
            "has_extended_info": True,
            "iso_3166_2": "US-CA"
        }]
        response.text.return_value = orjson.dumps(station_data).decode()
        
        stations = await radio_browser.search()
        
        assert len(stations) == 1
        session.request.assert_awaited_once()
        call_args = session.request.call_args
        assert "stations/search" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_station(self, radio_browser, mock_session):
        """Test station method."""
        session, response = mock_session
        station_data = [{
            "changeuuid": "change-uuid",
            "stationuuid": "test-uuid",
            "name": "Test Station",
            "url": "http://example.com/stream",
            "url_resolved": "http://example.com/stream",
            "homepage": "http://example.com",
            "favicon": "http://example.com/favicon.ico",
            "tags": "tag1,tag2",
            "country": "United States",
            "countrycode": "US",
            "state": "California",
            "language": "English",
            "languagecodes": "en",
            "votes": 10,
            "clickcount": 100,
            "clicktrend": 5,
            "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
            "codec": "MP3",
            "bitrate": 128,
            "hls": False,
            "lastcheckok": True,
            "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
            "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
            "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
            "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
            "ssl_error": 0,
            "geo_lat": 37.7749,
            "geo_long": -122.4194,
            "has_extended_info": True,
            "iso_3166_2": "US-CA"
        }]
        response.text.return_value = orjson.dumps(station_data).decode()
        
        station = await radio_browser.station(uuid="test-uuid")
        
        assert station is not None
        assert isinstance(station, Station)
        assert station.uuid == "test-uuid"
        assert station.name == "Test Station"
        
        session.request.assert_awaited_once()
        call_args = session.request.call_args
        assert "stations/byuuid/test-uuid" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_station_not_found(self, radio_browser, mock_session):
        """Test station method when station is not found."""
        _, response = mock_session
        response.text.return_value = "[]"
        
        station = await radio_browser.station(uuid="nonexistent-uuid")
        
        assert station is None

    @pytest.mark.asyncio
    async def test_stations(self, radio_browser, mock_session):
        """Test stations method."""
        session, response = mock_session
        station_data = [{
            "changeuuid": "change-uuid",
            "stationuuid": "station-uuid",
            "name": "Test Station",
            "url": "http://example.com/stream",
            "url_resolved": "http://example.com/stream",
            "homepage": "http://example.com",
            "favicon": "http://example.com/favicon.ico",
            "tags": "tag1,tag2",
            "country": "United States",
            "countrycode": "US",
            "state": "California",
            "language": "English",
            "languagecodes": "en",
            "votes": 10,
            "clickcount": 100,
            "clicktrend": 5,
            "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
            "codec": "MP3",
            "bitrate": 128,
            "hls": False,
            "lastcheckok": True,
            "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
            "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
            "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
            "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
            "ssl_error": 0,
            "geo_lat": 37.7749,
            "geo_long": -122.4194,
            "has_extended_info": True,
            "iso_3166_2": "US-CA"
        }]
        response.text.return_value = orjson.dumps(station_data).decode()
        
        stations = await radio_browser.stations(
            filter_by=FilterBy.COUNTRY,
            filter_term="US",
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.BITRATE,
            reverse=True
        )
        
        assert len(stations) == 1
        assert isinstance(stations[0], Station)
        assert stations[0].name == "Test Station"
        
        session.request.assert_awaited_once()
        call_args = session.request.call_args
        assert "stations/bycountry/US" in str(call_args[0][1])
        params = call_args[1]["params"]
        assert params["hidebroken"] == "true"
        assert params["limit"] == 10
        assert params["offset"] == 5
        assert params["order"] == "bitrate"
        assert params["reverse"] == "true"

    @pytest.mark.asyncio
    async def test_stations_without_filter(self, radio_browser, mock_session):
        """Test stations method without filter parameters."""
        session, response = mock_session
        station_data = [{
            "changeuuid": "change-uuid",
            "stationuuid": "station-uuid",
            "name": "Test Station",
            "url": "http://example.com/stream",
            "url_resolved": "http://example.com/stream",
            "homepage": "http://example.com",
            "favicon": "http://example.com/favicon.ico",
            "tags": "tag1,tag2",
            "country": "United States",
            "countrycode": "US",
            "state": "California",
            "language": "English",
            "languagecodes": "en",
            "votes": 10,
            "clickcount": 100,
            "clicktrend": 5,
            "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
            "codec": "MP3",
            "bitrate": 128,
            "hls": False,
            "lastcheckok": True,
            "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
            "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
            "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
            "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
            "ssl_error": 0,
            "geo_lat": 37.7749,
            "geo_long": -122.4194,
            "has_extended_info": True,
            "iso_3166_2": "US-CA"
        }]
        response.text.return_value = orjson.dumps(station_data).decode()
        
        stations = await radio_browser.stations()
        
        assert len(stations) == 1
        session.request.assert_awaited_once()
        call_args = session.request.call_args
        assert "stations" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_tags(self, radio_browser, mock_session):
        """Test tags method."""
        session, response = mock_session
        tags_data = [
            {"name": "rock", "stationcount": "100"},
            {"name": "pop", "stationcount": "50"},
            {"name": "jazz", "stationcount": "25"}
        ]
        response.text.return_value = orjson.dumps(tags_data).decode()
        
        tags = await radio_browser.tags(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True
        )
        
        assert len(tags) == 3
        assert isinstance(tags[0], Tag)
        assert tags[0].name == "rock"
        assert tags[0].station_count == "100"
        
        session.request.assert_awaited_once()
        call_args = session.request.call_args
        assert "tags" in str(call_args[0][1])
        params = call_args[1]["params"]
        assert params["hidebroken"] == "true"
        assert params["limit"] == 10
        assert params["offset"] == 5
        assert params["order"] == "stationcount"
        assert params["reverse"] == "true"

    @pytest.mark.asyncio
    async def test_close(self, mock_session):
        """Test close method."""
        session, _ = mock_session
        
        # Test with _close_session=True
        browser = RadioBrowser(user_agent="test_agent")
        browser.session = session
        browser._close_session = True
        
        await browser.close()
        
        session.close.assert_awaited_once()
        
        # Test with _close_session=False
        session.close.reset_mock()
        browser._close_session = False
        
        await browser.close()
        
        session.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_aenter(self):
        """Test __aenter__ method."""
        browser = RadioBrowser(user_agent="test_agent")
        result = await browser.__aenter__()
        assert result is browser

    @pytest.mark.asyncio
    async def test_aexit(self, mock_session):
        """Test __aexit__ method."""
        session, _ = mock_session
        
        browser = RadioBrowser(user_agent="test_agent")
        browser.session = session
        browser._close_session = True
        
        with patch.object(browser, "close", AsyncMock()) as mock_close:
            await browser.__aexit__(None, None, None)
            mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_session):
        """Test using RadioBrowser as an async context manager."""
        session, response = mock_session
        response.text.return_value = "{}"
        
        async with RadioBrowser(user_agent="test_agent") as browser:
            browser.session = session
            browser._host = "api.radio-browser.info"
            await browser.stats()
            
        session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_station_country_property(self, radio_browser, mock_session):
        """Test Station.country property."""
        _, response = mock_session
        station_data = [{
            "changeuuid": "change-uuid",
            "stationuuid": "station-uuid",
            "name": "Test Station",
            "url": "http://example.com/stream",
            "url_resolved": "http://example.com/stream",
            "homepage": "http://example.com",
            "favicon": "http://example.com/favicon.ico",
            "tags": "tag1,tag2",
            "country": "United States",
            "countrycode": "US",
            "state": "California",
            "language": "English",
            "languagecodes": "en",
            "votes": 10,
            "clickcount": 100,
            "clicktrend": 5,
            "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
            "codec": "MP3",
            "bitrate": 128,
            "hls": False,
            "lastcheckok": True,
            "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
            "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
            "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
            "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
            "ssl_error": 0,
            "geo_lat": 37.7749,
            "geo_long": -122.4194,
            "has_extended_info": True,
            "iso_3166_2": "US-CA"
        }]
        response.text.return_value = orjson.dumps(station_data).decode()
        
        stations = await radio_browser.stations()
        
        assert stations[0].country == "United States"
        
        # Test with invalid country code
        stations[0].country_code = "XX"
        assert stations[0].country is None

    @pytest.mark.asyncio
    async def test_request_creates_session(self, mock_dns_resolver):
        """Test _request creates a session if none exists."""
        with patch("aiohttp.ClientSession", autospec=True) as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.text = AsyncMock(return_value='{"key": "value"}')
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session_class.return_value = mock_session
            
            browser = RadioBrowser(user_agent="test_agent")
            browser._host = "api.radio-browser.info"  # Skip DNS resolution
            
            # Session should be None initially
            assert browser.session is None
            
            # Call _request
            await browser._request("test")
            
            # Session should be created
            assert browser.session is not None
            assert browser._close_session is True
            
            # Verify the request was made
            mock_session.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_countries_order_by_name(self, radio_browser, mock_session):
        """Test countries method with order by name."""
        _, response = mock_session
        # Deliberately out of order to test sorting
        countries_data = [
            {"name": "GB", "stationcount": "50"},
            {"name": "US", "stationcount": "100"},
            {"name": "CA", "stationcount": "30"},
        ]
        response.text.return_value = orjson.dumps(countries_data).decode()
        
        countries = await radio_browser.countries(order=Order.NAME)
        
        # Should be sorted by resolved name
        assert countries[0].name == "Canada"  # CA
        assert countries[1].name == "United Kingdom"  # GB
        assert countries[2].name == "United States"  # US
        
        # Test with a different order
        countries = await radio_browser.countries(order=Order.STATION_COUNT)
        
        # Should not be sorted by name in this case
        assert len(countries) == 3  # Just verify we got all countries

    @pytest.mark.asyncio
    async def test_search_with_filter_by_only(self, radio_browser, mock_session):
        """Test search method with only filter_by parameter."""
        session, response = mock_session
        response.text.return_value = "[]"
        
        await radio_browser.search(filter_by=FilterBy.COUNTRY)
        
        session.request.assert_awaited_once()
        call_args = session.request.call_args
        assert "stations/search/bycountry" in str(call_args[0][1])