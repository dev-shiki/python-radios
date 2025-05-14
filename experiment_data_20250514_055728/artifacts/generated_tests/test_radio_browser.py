import asyncio
import socket
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import orjson
import pytest
from aiodns.error import DNSError
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
def radio_browser():
    """Return a RadioBrowser instance."""
    return RadioBrowser(user_agent="Test/1.0")


@pytest.fixture
def mock_session():
    """Return a mock aiohttp ClientSession."""
    session = AsyncMock()
    session.request = AsyncMock()
    return session


@pytest.fixture
def mock_response():
    """Return a mock response."""
    response = AsyncMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    response.text = AsyncMock(return_value='{"key": "value"}')
    return response


@pytest.fixture
def mock_dns_resolver():
    """Return a mock DNSResolver."""
    resolver = AsyncMock()
    resolver.query = AsyncMock()
    result = MagicMock()
    result.host = "api.radio-browser.info"
    resolver.query.return_value = [result]
    return resolver


@pytest.fixture
def stats_data():
    """Return stats data."""
    return {
        "supported_version": 1,
        "software_version": "1.0.0",
        "status": "OK",
        "stations": 1000,
        "stations_broken": 10,
        "tags": 500,
        "clicks_last_hour": 100,
        "clicks_last_day": 1000,
        "languages": 50,
        "countries": 100,
    }


@pytest.fixture
def country_data():
    """Return country data."""
    return [
        {
            "name": "US",
            "stationcount": "100",
        },
        {
            "name": "GB",
            "stationcount": "50",
        },
    ]


@pytest.fixture
def language_data():
    """Return language data."""
    return [
        {
            "name": "english",
            "stationcount": "100",
            "iso_639": "en",
        },
        {
            "name": "german",
            "stationcount": "50",
            "iso_639": "de",
        },
    ]


@pytest.fixture
def station_data():
    """Return station data."""
    return [
        {
            "changeuuid": "12345",
            "stationuuid": "abcde",
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
            "lastchangetime_iso8601": "2021-01-01T00:00:00Z",
            "codec": "MP3",
            "bitrate": 128,
            "hls": False,
            "lastcheckok": True,
            "lastchecktime_iso8601": "2021-01-01T00:00:00Z",
            "lastcheckoktime_iso8601": "2021-01-01T00:00:00Z",
            "lastlocalchecktime_iso8601": "2021-01-01T00:00:00Z",
            "clicktimestamp_iso8601": "2021-01-01T00:00:00Z",
            "clickcount": 100,
            "clicktrend": 1,
            "ssl_error": 0,
            "geo_lat": 37.7749,
            "geo_long": -122.4194,
            "has_extended_info": False,
            "iso_3166_2": "US-CA",
        }
    ]


@pytest.fixture
def tag_data():
    """Return tag data."""
    return [
        {
            "name": "rock",
            "stationcount": "100",
        },
        {
            "name": "pop",
            "stationcount": "50",
        },
    ]


class TestRadioBrowser:
    """Tests for the RadioBrowser class."""

    @pytest.mark.asyncio
    async def test_init(self, radio_browser):
        """Test initialization of RadioBrowser."""
        assert radio_browser.user_agent == "Test/1.0"
        assert radio_browser.request_timeout == 8.0
        assert radio_browser.session is None
        assert radio_browser._close_session is False
        assert radio_browser._host is None

    @pytest.mark.asyncio
    @patch("radios.radio_browser.DNSResolver")
    async def test_request_dns_lookup(self, mock_dns_resolver_class, radio_browser, mock_session, mock_response):
        """Test _request method performs DNS lookup when host is None."""
        resolver = AsyncMock()
        mock_dns_resolver_class.return_value = resolver
        
        result = MagicMock()
        result.host = "api.radio-browser.info"
        resolver.query.return_value = [result]
        
        radio_browser.session = mock_session
        mock_session.request.return_value = mock_response
        
        await radio_browser._request("test")
        
        resolver.query.assert_called_once_with("_api._tcp.radio-browser.info", "SRV")
        assert radio_browser._host == "api.radio-browser.info"

    @pytest.mark.asyncio
    async def test_request_creates_session(self, radio_browser, mock_response):
        """Test _request creates a session if none exists."""
        with patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            mock_session.request.return_value = mock_response
            
            with patch("radios.radio_browser.DNSResolver") as mock_dns_resolver_class:
                resolver = AsyncMock()
                mock_dns_resolver_class.return_value = resolver
                
                result = MagicMock()
                result.host = "api.radio-browser.info"
                resolver.query.return_value = [result]
                
                await radio_browser._request("test")
                
                mock_session_class.assert_called_once()
                assert radio_browser.session is mock_session
                assert radio_browser._close_session is True

    @pytest.mark.asyncio
    async def test_request_timeout(self, radio_browser, mock_session):
        """Test _request handles timeout errors."""
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        
        mock_session.request.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(RadioBrowserConnectionTimeoutError):
            await radio_browser._request("test")
        
        assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_connection_error(self, radio_browser, mock_session):
        """Test _request handles connection errors."""
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        
        mock_session.request.side_effect = aiohttp.ClientError()
        
        with pytest.raises(RadioBrowserConnectionError):
            await radio_browser._request("test")
        
        assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_socket_error(self, radio_browser, mock_session):
        """Test _request handles socket errors."""
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        
        mock_session.request.side_effect = socket.gaierror()
        
        with pytest.raises(RadioBrowserConnectionError):
            await radio_browser._request("test")
        
        assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_invalid_content_type(self, radio_browser, mock_session):
        """Test _request handles invalid content type."""
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        
        response = AsyncMock()
        response.status = 200
        response.headers = {"Content-Type": "text/html"}
        response.text = AsyncMock(return_value="<html></html>")
        
        mock_session.request.return_value = response
        
        with pytest.raises(RadioBrowserError):
            await radio_browser._request("test")

    @pytest.mark.asyncio
    async def test_stats(self, radio_browser, stats_data):
        """Test stats method."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(stats_data)
            
            result = await radio_browser.stats()
            
            mock_request.assert_called_once_with("stats")
            assert isinstance(result, Stats)
            assert result.supported_version == stats_data["supported_version"]
            assert result.software_version == AwesomeVersion(stats_data["software_version"])
            assert result.status == stats_data["status"]
            assert result.stations == stats_data["stations"]
            assert result.stations_broken == stats_data["stations_broken"]
            assert result.tags == stats_data["tags"]
            assert result.clicks_last_hour == stats_data["clicks_last_hour"]
            assert result.clicks_last_day == stats_data["clicks_last_day"]
            assert result.languages == stats_data["languages"]
            assert result.countries == stats_data["countries"]

    @pytest.mark.asyncio
    async def test_station_click(self, radio_browser):
        """Test station_click method."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = "{}"
            
            await radio_browser.station_click(uuid="test-uuid")
            
            mock_request.assert_called_once_with("url/test-uuid")

    @pytest.mark.asyncio
    async def test_countries(self, radio_browser, country_data):
        """Test countries method."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(country_data)
            
            result = await radio_browser.countries()
            
            mock_request.assert_called_once_with(
                "countrycodes",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                }
            )
            assert isinstance(result, list)
            assert all(isinstance(country, Country) for country in result)
            assert len(result) == len(country_data)

    @pytest.mark.asyncio
    async def test_countries_with_params(self, radio_browser, country_data):
        """Test countries method with parameters."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(country_data)
            
            result = await radio_browser.countries(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.STATION_COUNT,
                reverse=True,
            )
            
            mock_request.assert_called_once_with(
                "countrycodes",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "stationcount",
                    "reverse": True,
                }
            )
            assert isinstance(result, list)
            assert all(isinstance(country, Country) for country in result)
            assert len(result) == len(country_data)

    @pytest.mark.asyncio
    async def test_countries_kosovo_special_case(self, radio_browser):
        """Test countries method handles Kosovo special case."""
        kosovo_data = [{"name": "XK", "stationcount": "10"}]
        
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(kosovo_data)
            
            result = await radio_browser.countries()
            
            assert len(result) == 1
            assert result[0].name == "Kosovo"
            assert result[0].code == "XK"

    @pytest.mark.asyncio
    async def test_languages(self, radio_browser, language_data):
        """Test languages method."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(language_data)
            
            result = await radio_browser.languages()
            
            mock_request.assert_called_once_with(
                "languages",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                }
            )
            assert isinstance(result, list)
            assert all(isinstance(language, Language) for language in result)
            assert len(result) == len(language_data)
            # Check title case conversion
            assert result[0].name == "English"
            assert result[1].name == "German"

    @pytest.mark.asyncio
    async def test_languages_with_params(self, radio_browser, language_data):
        """Test languages method with parameters."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(language_data)
            
            result = await radio_browser.languages(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.STATION_COUNT,
                reverse=True,
            )
            
            mock_request.assert_called_once_with(
                "languages",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "stationcount",
                    "reverse": True,
                }
            )
            assert isinstance(result, list)
            assert all(isinstance(language, Language) for language in result)
            assert len(result) == len(language_data)

    @pytest.mark.asyncio
    async def test_search(self, radio_browser, station_data):
        """Test search method."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(station_data)
            
            result = await radio_browser.search()
            
            mock_request.assert_called_once_with(
                "stations/search",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                    "name": None,
                    "name_exact": False,
                    "country": "",
                    "country_exact": False,
                    "state_exact": False,
                    "language_exact": False,
                    "tag_exact": False,
                    "bitrate_min": 0,
                    "bitrate_max": 1000000,
                }
            )
            assert isinstance(result, list)
            assert all(isinstance(station, Station) for station in result)
            assert len(result) == len(station_data)

    @pytest.mark.asyncio
    async def test_search_with_filter(self, radio_browser, station_data):
        """Test search method with filter."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(station_data)
            
            result = await radio_browser.search(
                filter_by=FilterBy.NAME,
                filter_term="test",
            )
            
            mock_request.assert_called_once_with(
                "stations/search/byname/test",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                    "name": None,
                    "name_exact": False,
                    "country": "",
                    "country_exact": False,
                    "state_exact": False,
                    "language_exact": False,
                    "tag_exact": False,
                    "bitrate_min": 0,
                    "bitrate_max": 1000000,
                }
            )
            assert isinstance(result, list)
            assert all(isinstance(station, Station) for station in result)

    @pytest.mark.asyncio
    async def test_search_with_params(self, radio_browser, station_data):
        """Test search method with parameters."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(station_data)
            
            result = await radio_browser.search(
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
            
            mock_request.assert_called_once_with(
                "stations/search",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "bitrate",
                    "reverse": True,
                    "name": "test",
                    "name_exact": True,
                    "country": "US",
                    "country_exact": True,
                    "state_exact": True,
                    "language_exact": True,
                    "tag_exact": True,
                    "bitrate_min": 128,
                    "bitrate_max": 320,
                }
            )
            assert isinstance(result, list)
            assert all(isinstance(station, Station) for station in result)

    @pytest.mark.asyncio
    async def test_station(self, radio_browser, station_data):
        """Test station method."""
        with patch.object(radio_browser, "stations") as mock_stations:
            station = Station.from_dict(station_data[0])
            mock_stations.return_value = [station]
            
            result = await radio_browser.station(uuid="test-uuid")
            
            mock_stations.assert_called_once_with(
                filter_by=FilterBy.UUID,
                filter_term="test-uuid",
                limit=1,
            )
            assert result == station

    @pytest.mark.asyncio
    async def test_station_not_found(self, radio_browser):
        """Test station method when station is not found."""
        with patch.object(radio_browser, "stations") as mock_stations:
            mock_stations.return_value = []
            
            result = await radio_browser.station(uuid="test-uuid")
            
            mock_stations.assert_called_once_with(
                filter_by=FilterBy.UUID,
                filter_term="test-uuid",
                limit=1,
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_stations(self, radio_browser, station_data):
        """Test stations method."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(station_data)
            
            result = await radio_browser.stations()
            
            mock_request.assert_called_once_with(
                "stations",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                }
            )
            assert isinstance(result, list)
            assert all(isinstance(station, Station) for station in result)
            assert len(result) == len(station_data)

    @pytest.mark.asyncio
    async def test_stations_with_filter(self, radio_browser, station_data):
        """Test stations method with filter."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(station_data)
            
            result = await radio_browser.stations(
                filter_by=FilterBy.NAME,
                filter_term="test",
            )
            
            mock_request.assert_called_once_with(
                "stations/byname/test",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                }
            )
            assert isinstance(result, list)
            assert all(isinstance(station, Station) for station in result)

    @pytest.mark.asyncio
    async def test_stations_with_params(self, radio_browser, station_data):
        """Test stations method with parameters."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(station_data)
            
            result = await radio_browser.stations(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.BITRATE,
                reverse=True,
            )
            
            mock_request.assert_called_once_with(
                "stations",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "bitrate",
                    "reverse": True,
                }
            )
            assert isinstance(result, list)
            assert all(isinstance(station, Station) for station in result)

    @pytest.mark.asyncio
    async def test_tags(self, radio_browser, tag_data):
        """Test tags method."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(tag_data)
            
            result = await radio_browser.tags()
            
            mock_request.assert_called_once_with(
                "tags",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                }
            )
            assert isinstance(result, list)
            assert all(isinstance(tag, Tag) for tag in result)
            assert len(result) == len(tag_data)

    @pytest.mark.asyncio
    async def test_tags_with_params(self, radio_browser, tag_data):
        """Test tags method with parameters."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(tag_data)
            
            result = await radio_browser.tags(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.STATION_COUNT,
                reverse=True,
            )
            
            mock_request.assert_called_once_with(
                "tags",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "stationcount",
                    "reverse": True,
                }
            )
            assert isinstance(result, list)
            assert all(isinstance(tag, Tag) for tag in result)

    @pytest.mark.asyncio
    async def test_close(self, radio_browser, mock_session):
        """Test close method."""
        radio_browser.session = mock_session
        radio_browser._close_session = True
        
        await radio_browser.close()
        
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_session(self, radio_browser):
        """Test close method when no session exists."""
        radio_browser.session = None
        
        await radio_browser.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_not_owner(self, radio_browser, mock_session):
        """Test close method when not session owner."""
        radio_browser.session = mock_session
        radio_browser._close_session = False
        
        await radio_browser.close()
        
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_aenter(self, radio_browser):
        """Test __aenter__ method."""
        result = await radio_browser.__aenter__()
        assert result is radio_browser

    @pytest.mark.asyncio
    async def test_aexit(self, radio_browser):
        """Test __aexit__ method."""
        with patch.object(radio_browser, "close") as mock_close:
            await radio_browser.__aexit__(None, None, None)
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, radio_browser):
        """Test using RadioBrowser as a context manager."""
        with patch.object(radio_browser, "close") as mock_close:
            async with radio_browser as rb:
                assert rb is radio_browser
            mock_close.assert_called_once()