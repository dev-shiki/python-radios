import asyncio
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import orjson
import pytest
from aiodns.error import DNSError
from aiohttp import ClientError
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
    """Mock the DNSResolver."""
    with patch("radios.radio_browser.DNSResolver") as mock_resolver:
        resolver_instance = MagicMock()
        mock_resolver.return_value = resolver_instance
        
        # Create a mock SRV record
        mock_srv_record = MagicMock()
        mock_srv_record.host = "api.radio-browser.info"
        
        # Make query return an awaitable
        query_result = AsyncMock()
        query_result.__await__ = lambda: [mock_srv_record].__await__()
        resolver_instance.query = AsyncMock(return_value=[mock_srv_record])
        
        yield resolver_instance


@pytest.fixture
def mock_session():
    """Mock aiohttp ClientSession."""
    with patch("aiohttp.ClientSession") as mock_client:
        session_instance = AsyncMock()
        mock_client.return_value = session_instance
        
        # Mock the response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.text = AsyncMock()
        
        # Set up the session request method
        session_instance.request.return_value = mock_response
        
        yield session_instance, mock_response


@pytest.fixture
async def radio_browser():
    """Return a RadioBrowser instance."""
    browser = RadioBrowser(user_agent="Test/1.0")
    yield browser
    await browser.close()


class TestRadioBrowser:
    """Tests for the RadioBrowser class."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test initializing the API client."""
        client = RadioBrowser(user_agent="Test/1.0")
        assert client.user_agent == "Test/1.0"
        assert client.request_timeout == 8.0
        assert client.session is None
        assert client._close_session is False
        assert client._host is None
        await client.close()

    @pytest.mark.asyncio
    async def test_request_dns_resolution(self, mock_dns_resolver, mock_session):
        """Test DNS resolution in _request method."""
        session, response = mock_session
        response.text.return_value = "{}"
        
        client = RadioBrowser(user_agent="Test/1.0")
        await client._request("test")
        
        # Verify DNS resolution was performed
        mock_dns_resolver.query.assert_called_once_with(
            "_api._tcp.radio-browser.info", "SRV"
        )
        
        # Verify the host was set
        assert client._host == "api.radio-browser.info"
        
        # Verify the request was made with the correct parameters
        session.request.assert_called_once()
        call_args = session.request.call_args
        assert call_args[0][0] == "GET"  # Method
        assert "api.radio-browser.info" in str(call_args[0][1])  # URL
        assert call_args[1]["headers"]["User-Agent"] == "Test/1.0"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_request_session_creation(self, mock_dns_resolver, mock_session):
        """Test session creation in _request method."""
        session, response = mock_session
        response.text.return_value = "{}"
        
        client = RadioBrowser(user_agent="Test/1.0")
        assert client.session is None
        
        await client._request("test")
        
        assert client.session is not None
        assert client._close_session is True
        
        await client.close()

    @pytest.mark.asyncio
    async def test_request_with_params(self, mock_dns_resolver, mock_session):
        """Test _request method with parameters."""
        session, response = mock_session
        response.text.return_value = "{}"
        
        client = RadioBrowser(user_agent="Test/1.0")
        await client._request("test", params={"test": "value", "bool_param": True})
        
        call_args = session.request.call_args
        assert "params" in call_args[1]
        assert call_args[1]["params"]["test"] == "value"
        assert call_args[1]["params"]["bool_param"] == "true"  # Converted to lowercase string
        
        await client.close()

    @pytest.mark.asyncio
    async def test_request_timeout_error(self, mock_dns_resolver, mock_session):
        """Test timeout error in _request method."""
        session, _ = mock_session
        session.request.side_effect = asyncio.TimeoutError()
        
        client = RadioBrowser(user_agent="Test/1.0")
        
        with pytest.raises(RadioBrowserConnectionTimeoutError):
            await client._request("test")
        
        # Host should be reset on error
        assert client._host is None
        
        await client.close()

    @pytest.mark.asyncio
    async def test_request_client_error(self, mock_dns_resolver, mock_session):
        """Test client error in _request method."""
        session, _ = mock_session
        session.request.side_effect = ClientError()
        
        client = RadioBrowser(user_agent="Test/1.0")
        
        with pytest.raises(RadioBrowserConnectionError):
            await client._request("test")
        
        # Host should be reset on error
        assert client._host is None
        
        await client.close()

    @pytest.mark.asyncio
    async def test_request_socket_error(self, mock_dns_resolver, mock_session):
        """Test socket error in _request method."""
        session, _ = mock_session
        session.request.side_effect = socket.gaierror()
        
        client = RadioBrowser(user_agent="Test/1.0")
        
        with pytest.raises(RadioBrowserConnectionError):
            await client._request("test")
        
        # Host should be reset on error
        assert client._host is None
        
        await client.close()

    @pytest.mark.asyncio
    async def test_request_invalid_content_type(self, mock_dns_resolver, mock_session):
        """Test invalid content type in _request method."""
        session, response = mock_session
        response.headers = {"Content-Type": "text/html"}
        response.text.return_value = "<html>Not JSON</html>"
        response.status = 200
        
        client = RadioBrowser(user_agent="Test/1.0")
        
        with pytest.raises(RadioBrowserError):
            await client._request("test")
        
        await client.close()

    @pytest.mark.asyncio
    async def test_stats(self, mock_dns_resolver, mock_session):
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
        
        client = RadioBrowser(user_agent="Test/1.0")
        result = await client.stats()
        
        assert isinstance(result, Stats)
        assert result.supported_version == 1
        assert result.software_version == AwesomeVersion("2.0.0")
        assert result.status == "OK"
        assert result.stations == 1000
        assert result.stations_broken == 10
        assert result.tags == 500
        assert result.clicks_last_hour == 100
        assert result.clicks_last_day == 1000
        assert result.languages == 50
        assert result.countries == 100
        
        await client.close()

    @pytest.mark.asyncio
    async def test_station_click(self, mock_dns_resolver, mock_session):
        """Test station_click method."""
        session, response = mock_session
        response.text.return_value = "{}"
        
        client = RadioBrowser(user_agent="Test/1.0")
        await client.station_click(uuid="test-uuid")
        
        # Verify the request was made with the correct URL
        call_args = session.request.call_args
        assert "url/test-uuid" in str(call_args[0][1])
        
        await client.close()

    @pytest.mark.asyncio
    async def test_countries(self, mock_dns_resolver, mock_session):
        """Test countries method."""
        session, response = mock_session
        countries_data = [
            {"name": "US", "stationcount": "100"},
            {"name": "GB", "stationcount": "50"},
            {"name": "XK", "stationcount": "10"},  # Kosovo special case
        ]
        response.text.return_value = orjson.dumps(countries_data).decode()
        
        client = RadioBrowser(user_agent="Test/1.0")
        result = await client.countries()
        
        # Verify the request was made with the correct URL and parameters
        call_args = session.request.call_args
        assert "countrycodes" in str(call_args[0][1])
        assert call_args[1]["params"]["hidebroken"] == "false"
        assert call_args[1]["params"]["limit"] == 100000
        assert call_args[1]["params"]["offset"] == 0
        assert call_args[1]["params"]["order"] == "name"
        assert call_args[1]["params"]["reverse"] == "false"
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(country, Country) for country in result)
        
        # Check country name resolution
        assert any(country.name == "United States" for country in result)
        assert any(country.name == "United Kingdom" for country in result)
        assert any(country.name == "Kosovo" for country in result)  # Special case
        
        await client.close()

    @pytest.mark.asyncio
    async def test_countries_with_parameters(self, mock_dns_resolver, mock_session):
        """Test countries method with custom parameters."""
        session, response = mock_session
        countries_data = [{"name": "US", "stationcount": "100"}]
        response.text.return_value = orjson.dumps(countries_data).decode()
        
        client = RadioBrowser(user_agent="Test/1.0")
        await client.countries(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True,
        )
        
        # Verify the parameters were passed correctly
        call_args = session.request.call_args
        assert call_args[1]["params"]["hidebroken"] == "true"
        assert call_args[1]["params"]["limit"] == 10
        assert call_args[1]["params"]["offset"] == 5
        assert call_args[1]["params"]["order"] == "stationcount"
        assert call_args[1]["params"]["reverse"] == "true"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_languages(self, mock_dns_resolver, mock_session):
        """Test languages method."""
        session, response = mock_session
        languages_data = [
            {"name": "english", "stationcount": "100", "iso_639": "en"},
            {"name": "spanish", "stationcount": "50", "iso_639": "es"},
        ]
        response.text.return_value = orjson.dumps(languages_data).decode()
        
        client = RadioBrowser(user_agent="Test/1.0")
        result = await client.languages()
        
        # Verify the request was made with the correct URL and parameters
        call_args = session.request.call_args
        assert "languages" in str(call_args[0][1])
        assert call_args[1]["params"]["hidebroken"] == "false"
        assert call_args[1]["params"]["limit"] == 100000
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(language, Language) for language in result)
        
        # Check language name capitalization
        assert result[0].name == "English"
        assert result[1].name == "Spanish"
        
        # Check language codes
        assert result[0].code == "en"
        assert result[1].code == "es"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_languages_with_parameters(self, mock_dns_resolver, mock_session):
        """Test languages method with custom parameters."""
        session, response = mock_session
        languages_data = [{"name": "english", "stationcount": "100", "iso_639": "en"}]
        response.text.return_value = orjson.dumps(languages_data).decode()
        
        client = RadioBrowser(user_agent="Test/1.0")
        await client.languages(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True,
        )
        
        # Verify the parameters were passed correctly
        call_args = session.request.call_args
        assert call_args[1]["params"]["hidebroken"] == "true"
        assert call_args[1]["params"]["limit"] == 10
        assert call_args[1]["params"]["offset"] == 5
        assert call_args[1]["params"]["order"] == "stationcount"
        assert call_args[1]["params"]["reverse"] == "true"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_search(self, mock_dns_resolver, mock_session):
        """Test search method."""
        session, response = mock_session
        stations_data = [
            {
                "changeuuid": "uuid1",
                "stationuuid": "station1",
                "name": "Test Station 1",
                "url": "http://example.com/stream1",
                "url_resolved": "http://example.com/stream1",
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
                "bitrate": 128,
                "codec": "MP3",
                "lastcheckok": True,
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "hls": False,
                "iso_3166_2": "US-CA",
                "ssl_error": 0,
            }
        ]
        response.text.return_value = orjson.dumps(stations_data).decode()
        
        client = RadioBrowser(user_agent="Test/1.0")
        result = await client.search(name="test")
        
        # Verify the request was made with the correct URL and parameters
        call_args = session.request.call_args
        assert "stations/search" in str(call_args[0][1])
        assert call_args[1]["params"]["name"] == "test"
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(station, Station) for station in result)
        assert result[0].name == "Test Station 1"
        assert result[0].uuid == "station1"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_search_with_filter(self, mock_dns_resolver, mock_session):
        """Test search method with filter parameters."""
        session, response = mock_session
        stations_data = [{"stationuuid": "station1", "name": "Test Station", "bitrate": 128, "changeuuid": "uuid1", "clickcount": 100, "clicktrend": 5, "codec": "MP3", "countrycode": "US", "favicon": "http://example.com/favicon.ico", "geo_lat": 37.7749, "geo_long": -122.4194, "has_extended_info": True, "hls": False, "homepage": "http://example.com", "iso_3166_2": "US-CA", "language": "English", "languagecodes": "en", "lastchangetime_iso8601": "2023-01-01T00:00:00Z", "lastcheckok": True, "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z", "lastchecktime_iso8601": "2023-01-01T00:00:00Z", "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z", "clicktimestamp_iso8601": "2023-01-01T00:00:00Z", "ssl_error": 0, "state": "California", "tags": "tag1,tag2", "url": "http://example.com/stream", "url_resolved": "http://example.com/stream", "votes": 10}]
        response.text.return_value = orjson.dumps(stations_data).decode()
        
        client = RadioBrowser(user_agent="Test/1.0")
        await client.search(
            filter_by=FilterBy.COUNTRY,
            filter_term="US",
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.BITRATE,
            reverse=True,
            name="test",
            name_exact=True,
            country="US",
            country_exact=True,
            state_exact=True,
            language_exact=True,
            tag_exact=True,
            bitrate_min=128,
            bitrate_max=320,
        )
        
        # Verify the request was made with the correct URL
        call_args = session.request.call_args
        assert "stations/search/bycountry/US" in str(call_args[0][1])
        
        # Verify all parameters were passed correctly
        params = call_args[1]["params"]
        assert params["hidebroken"] == "true"
        assert params["limit"] == 10
        assert params["offset"] == 5
        assert params["order"] == "bitrate"
        assert params["reverse"] == "true"
        assert params["name"] == "test"
        assert params["name_exact"] == "true"
        assert params["country"] == "US"
        assert params["country_exact"] == "true"
        assert params["state_exact"] == "true"
        assert params["language_exact"] == "true"
        assert params["tag_exact"] == "true"
        assert params["bitrate_min"] == 128
        assert params["bitrate_max"] == 320
        
        await client.close()

    @pytest.mark.asyncio
    async def test_station(self, mock_dns_resolver, mock_session):
        """Test station method."""
        session, response = mock_session
        stations_data = [{"stationuuid": "test-uuid", "name": "Test Station", "bitrate": 128, "changeuuid": "uuid1", "clickcount": 100, "clicktrend": 5, "codec": "MP3", "countrycode": "US", "favicon": "http://example.com/favicon.ico", "geo_lat": 37.7749, "geo_long": -122.4194, "has_extended_info": True, "hls": False, "homepage": "http://example.com", "iso_3166_2": "US-CA", "language": "English", "languagecodes": "en", "lastchangetime_iso8601": "2023-01-01T00:00:00Z", "lastcheckok": True, "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z", "lastchecktime_iso8601": "2023-01-01T00:00:00Z", "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z", "clicktimestamp_iso8601": "2023-01-01T00:00:00Z", "ssl_error": 0, "state": "California", "tags": "tag1,tag2", "url": "http://example.com/stream", "url_resolved": "http://example.com/stream", "votes": 10}]
        response.text.return_value = orjson.dumps(stations_data).decode()
        
        client = RadioBrowser(user_agent="Test/1.0")
        result = await client.station(uuid="test-uuid")
        
        # Verify the request was made with the correct URL and parameters
        call_args = session.request.call_args
        assert "stations/byuuid/test-uuid" in str(call_args[0][1])
        
        # Verify the result
        assert isinstance(result, Station)
        assert result.uuid == "test-uuid"
        assert result.name == "Test Station"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_station_not_found(self, mock_dns_resolver, mock_session):
        """Test station method when station is not found."""
        _, response = mock_session
        response.text.return_value = "[]"  # Empty array
        
        client = RadioBrowser(user_agent="Test/1.0")
        result = await client.station(uuid="nonexistent-uuid")
        
        # Verify the result is None
        assert result is None
        
        await client.close()

    @pytest.mark.asyncio
    async def test_stations(self, mock_dns_resolver, mock_session):
        """Test stations method."""
        session, response = mock_session
        stations_data = [{"stationuuid": "station1", "name": "Test Station", "bitrate": 128, "changeuuid": "uuid1", "clickcount": 100, "clicktrend": 5, "codec": "MP3", "countrycode": "US", "favicon": "http://example.com/favicon.ico", "geo_lat": 37.7749, "geo_long": -122.4194, "has_extended_info": True, "hls": False, "homepage": "http://example.com", "iso_3166_2": "US-CA", "language": "English", "languagecodes": "en", "lastchangetime_iso8601": "2023-01-01T00:00:00Z", "lastcheckok": True, "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z", "lastchecktime_iso8601": "2023-01-01T00:00:00Z", "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z", "clicktimestamp_iso8601": "2023-01-01T00:00:00Z", "ssl_error": 0, "state": "California", "tags": "tag1,tag2", "url": "http://example.com/stream", "url_resolved": "http://example.com/stream", "votes": 10}]
        response.text.return_value = orjson.dumps(stations_data).decode()
        
        client = RadioBrowser(user_agent="Test/1.0")
        result = await client.stations()
        
        # Verify the request was made with the correct URL and parameters
        call_args = session.request.call_args
        assert "stations" in str(call_args[0][1])
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(station, Station) for station in result)
        
        await client.close()

    @pytest.mark.asyncio
    async def test_stations_with_filter(self, mock_dns_resolver, mock_session):
        """Test stations method with filter parameters."""
        session, response = mock_session
        stations_data = [{"stationuuid": "station1", "name": "Test Station", "bitrate": 128, "changeuuid": "uuid1", "clickcount": 100, "clicktrend": 5, "codec": "MP3", "countrycode": "US", "favicon": "http://example.com/favicon.ico", "geo_lat": 37.7749, "geo_long": -122.4194, "has_extended_info": True, "hls": False, "homepage": "http://example.com", "iso_3166_2": "US-CA", "language": "English", "languagecodes": "en", "lastchangetime_iso8601": "2023-01-01T00:00:00Z", "lastcheckok": True, "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z", "lastchecktime_iso8601": "2023-01-01T00:00:00Z", "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z", "clicktimestamp_iso8601": "2023-01-01T00:00:00Z", "ssl_error": 0, "state": "California", "tags": "tag1,tag2", "url": "http://example.com/stream", "url_resolved": "http://example.com/stream", "votes": 10}]
        response.text.return_value = orjson.dumps(stations_data).decode()
        
        client = RadioBrowser(user_agent="Test/1.0")
        await client.stations(
            filter_by=FilterBy.COUNTRY,
            filter_term="US",
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.BITRATE,
            reverse=True,
        )
        
        # Verify the request was made with the correct URL
        call_args = session.request.call_args
        assert "stations/bycountry/US" in str(call_args[0][1])
        
        # Verify all parameters were passed correctly
        params = call_args[1]["params"]
        assert params["hidebroken"] == "true"
        assert params["limit"] == 10
        assert params["offset"] == 5
        assert params["order"] == "bitrate"
        assert params["reverse"] == "true"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_tags(self, mock_dns_resolver, mock_session):
        """Test tags method."""
        session, response = mock_session
        tags_data = [
            {"name": "rock", "stationcount": "100"},
            {"name": "pop", "stationcount": "50"},
        ]
        response.text.return_value = orjson.dumps(tags_data).decode()
        
        client = RadioBrowser(user_agent="Test/1.0")
        result = await client.tags()
        
        # Verify the request was made with the correct URL and parameters
        call_args = session.request.call_args
        assert "tags" in str(call_args[0][1])
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(tag, Tag) for tag in result)
        assert result[0].name == "rock"
        assert result[0].station_count == "100"
        assert result[1].name == "pop"
        assert result[1].station_count == "50"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_tags_with_parameters(self, mock_dns_resolver, mock_session):
        """Test tags method with custom parameters."""
        session, response = mock_session
        tags_data = [{"name": "rock", "stationcount": "100"}]
        response.text.return_value = orjson.dumps(tags_data).decode()
        
        client = RadioBrowser(user_agent="Test/1.0")
        await client.tags(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True,
        )
        
        # Verify the parameters were passed correctly
        call_args = session.request.call_args
        assert call_args[1]["params"]["hidebroken"] == "true"
        assert call_args[1]["params"]["limit"] == 10
        assert call_args[1]["params"]["offset"] == 5
        assert call_args[1]["params"]["order"] == "stationcount"
        assert call_args[1]["params"]["reverse"] == "true"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_close(self):
        """Test close method."""
        with patch("aiohttp.ClientSession.close") as mock_close:
            mock_close.return_value = asyncio.Future()
            mock_close.return_value.set_result(None)
            
            client = RadioBrowser(user_agent="Test/1.0")
            client.session = aiohttp.ClientSession()
            client._close_session = True
            
            await client.close()
            
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_session(self):
        """Test close method when no session exists."""
        client = RadioBrowser(user_agent="Test/1.0")
        client.session = None
        
        # Should not raise an exception
        await client.close()

    @pytest.mark.asyncio
    async def test_close_not_owned_session(self):
        """Test close method when session is not owned."""
        with patch("aiohttp.ClientSession.close") as mock_close:
            client = RadioBrowser(user_agent="Test/1.0")
            client.session = aiohttp.ClientSession()
            client._close_session = False
            
            await client.close()
            
            # Should not close the session
            mock_close.assert_not_called()
            
            # Clean up
            await client.session.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_dns_resolver, mock_session):
        """Test using RadioBrowser as an async context manager."""
        with patch.object(RadioBrowser, "close") as mock_close:
            mock_close.return_value = asyncio.Future()
            mock_close.return_value.set_result(None)
            
            async with RadioBrowser(user_agent="Test/1.0") as client:
                assert isinstance(client, RadioBrowser)
                assert client.user_agent == "Test/1.0"
            
            # Verify close was called
            mock_close.assert_called_once()