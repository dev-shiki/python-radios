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
    """Return a mock DNS resolver."""
    resolver = AsyncMock()
    result = [MagicMock()]
    result[0].host = "api.radio-browser.info"
    resolver.query.return_value = result
    return resolver


class TestRadioBrowser:
    """Tests for the RadioBrowser class."""

    @pytest.mark.asyncio
    async def test_request_dns_resolution(self, radio_browser, mock_session, mock_response, mock_dns_resolver):
        """Test that _request resolves DNS when host is None."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            with patch.object(radio_browser, "session", mock_session):
                mock_session.request.return_value = mock_response
                
                await radio_browser._request("test")
                
                mock_dns_resolver.query.assert_called_once_with(
                    "_api._tcp.radio-browser.info", "SRV"
                )
                assert radio_browser._host == "api.radio-browser.info"

    @pytest.mark.asyncio
    async def test_request_creates_session_if_none(self, radio_browser, mock_response):
        """Test that _request creates a session if none exists."""
        with patch("radios.radio_browser.DNSResolver") as mock_resolver_cls:
            mock_resolver = AsyncMock()
            result = [MagicMock()]
            result[0].host = "api.radio-browser.info"
            mock_resolver.query.return_value = result
            mock_resolver_cls.return_value = mock_resolver
            
            with patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_cls:
                mock_session = AsyncMock()
                mock_session.request.return_value = mock_response
                mock_session_cls.return_value = mock_session
                
                await radio_browser._request("test")
                
                mock_session_cls.assert_called_once()
                assert radio_browser.session is not None
                assert radio_browser._close_session is True

    @pytest.mark.asyncio
    async def test_request_timeout(self, radio_browser, mock_session, mock_dns_resolver):
        """Test that _request raises RadioBrowserConnectionTimeoutError on timeout."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            with patch.object(radio_browser, "session", mock_session):
                mock_session.request.side_effect = asyncio.TimeoutError()
                
                with pytest.raises(RadioBrowserConnectionTimeoutError):
                    await radio_browser._request("test")
                
                assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_connection_error(self, radio_browser, mock_session, mock_dns_resolver):
        """Test that _request raises RadioBrowserConnectionError on connection error."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            with patch.object(radio_browser, "session", mock_session):
                mock_session.request.side_effect = aiohttp.ClientError()
                
                with pytest.raises(RadioBrowserConnectionError):
                    await radio_browser._request("test")
                
                assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_socket_error(self, radio_browser, mock_session, mock_dns_resolver):
        """Test that _request raises RadioBrowserConnectionError on socket error."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            with patch.object(radio_browser, "session", mock_session):
                mock_session.request.side_effect = socket.gaierror()
                
                with pytest.raises(RadioBrowserConnectionError):
                    await radio_browser._request("test")
                
                assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_invalid_content_type(self, radio_browser, mock_session, mock_response, mock_dns_resolver):
        """Test that _request raises RadioBrowserError on invalid content type."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            with patch.object(radio_browser, "session", mock_session):
                mock_response.headers = {"Content-Type": "text/html"}
                mock_session.request.return_value = mock_response
                
                with pytest.raises(RadioBrowserError):
                    await radio_browser._request("test")

    @pytest.mark.asyncio
    async def test_request_boolean_params(self, radio_browser, mock_session, mock_response, mock_dns_resolver):
        """Test that _request converts boolean params to lowercase strings."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            with patch.object(radio_browser, "session", mock_session):
                mock_session.request.return_value = mock_response
                
                await radio_browser._request("test", params={"bool_param": True})
                
                mock_session.request.assert_called_once()
                call_args = mock_session.request.call_args[1]
                assert call_args["params"] == {"bool_param": "true"}

    @pytest.mark.asyncio
    async def test_stats(self, radio_browser):
        """Test stats method returns Stats object."""
        stats_json = """
        {
            "supported_version": 1,
            "software_version": "1.0.0",
            "status": "OK",
            "stations": 1000,
            "stations_broken": 10,
            "tags": 500,
            "clicks_last_hour": 100,
            "clicks_last_day": 1000,
            "languages": 50,
            "countries": 100
        }
        """
        
        with patch.object(radio_browser, "_request", AsyncMock(return_value=stats_json)):
            result = await radio_browser.stats()
            
            assert isinstance(result, Stats)
            assert result.supported_version == 1
            assert result.software_version == AwesomeVersion("1.0.0")
            assert result.status == "OK"
            assert result.stations == 1000
            assert result.stations_broken == 10
            assert result.tags == 500
            assert result.clicks_last_hour == 100
            assert result.clicks_last_day == 1000
            assert result.languages == 50
            assert result.countries == 100

    @pytest.mark.asyncio
    async def test_station_click(self, radio_browser):
        """Test station_click method calls correct endpoint."""
        with patch.object(radio_browser, "_request", AsyncMock()) as mock_request:
            await radio_browser.station_click(uuid="test-uuid")
            
            mock_request.assert_called_once_with("url/test-uuid")

    @pytest.mark.asyncio
    async def test_countries(self, radio_browser):
        """Test countries method returns list of Country objects."""
        countries_json = """
        [
            {
                "name": "US",
                "stationcount": "100"
            },
            {
                "name": "XK",
                "stationcount": "50"
            }
        ]
        """
        
        with patch.object(radio_browser, "_request", AsyncMock(return_value=countries_json)):
            result = await radio_browser.countries()
            
            assert len(result) == 2
            assert isinstance(result[0], Country)
            assert result[0].code == "US"
            assert result[0].name == "United States"
            assert result[0].station_count == "100"
            
            assert isinstance(result[1], Country)
            assert result[1].code == "XK"
            assert result[1].name == "Kosovo"
            assert result[1].station_count == "50"

    @pytest.mark.asyncio
    async def test_countries_with_parameters(self, radio_browser):
        """Test countries method with parameters."""
        with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")) as mock_request:
            await radio_browser.countries(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.NAME,
                reverse=True
            )
            
            mock_request.assert_called_once()
            _, kwargs = mock_request.call_args
            assert kwargs["params"] == {
                "hidebroken": "true",
                "limit": 10,
                "offset": 5,
                "order": "name",
                "reverse": "true"
            }

    @pytest.mark.asyncio
    async def test_languages(self, radio_browser):
        """Test languages method returns list of Language objects."""
        languages_json = """
        [
            {
                "name": "english",
                "iso_639": "en",
                "stationcount": "500"
            },
            {
                "name": "spanish",
                "iso_639": "es",
                "stationcount": "300"
            }
        ]
        """
        
        with patch.object(radio_browser, "_request", AsyncMock(return_value=languages_json)):
            result = await radio_browser.languages()
            
            assert len(result) == 2
            assert isinstance(result[0], Language)
            assert result[0].code == "en"
            assert result[0].name == "English"
            assert result[0].station_count == "500"
            
            assert isinstance(result[1], Language)
            assert result[1].code == "es"
            assert result[1].name == "Spanish"
            assert result[1].station_count == "300"

    @pytest.mark.asyncio
    async def test_languages_with_parameters(self, radio_browser):
        """Test languages method with parameters."""
        with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")) as mock_request:
            await radio_browser.languages(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.STATION_COUNT,
                reverse=True
            )
            
            mock_request.assert_called_once()
            _, kwargs = mock_request.call_args
            assert kwargs["params"] == {
                "hidebroken": "true",
                "limit": 10,
                "offset": 5,
                "order": "stationcount",
                "reverse": "true"
            }

    @pytest.mark.asyncio
    async def test_search(self, radio_browser):
        """Test search method returns list of Station objects."""
        stations_json = """
        [
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
                "language": "English",
                "languagecodes": "en",
                "votes": 10,
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "codec": "MP3",
                "bitrate": 128,
                "hls": false,
                "lastcheckok": true,
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clickcount": 100,
                "clicktrend": 5,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": true,
                "iso_3166_2": "US-CA"
            }
        ]
        """
        
        with patch.object(radio_browser, "_request", AsyncMock(return_value=stations_json)):
            result = await radio_browser.search(
                name="test",
                country="US",
                bitrate_min=128
            )
            
            assert len(result) == 1
            assert isinstance(result[0], Station)
            assert result[0].uuid == "station-uuid-1"
            assert result[0].name == "Test Station 1"
            assert result[0].url == "http://example.com/stream1"
            assert result[0].bitrate == 128
            assert result[0].country_code == "US"

    @pytest.mark.asyncio
    async def test_search_with_filter(self, radio_browser):
        """Test search method with filter parameters."""
        with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")) as mock_request:
            await radio_browser.search(
                filter_by=FilterBy.NAME,
                filter_term="test",
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.BITRATE,
                reverse=True,
                name_exact=True,
                country_exact=True,
                state_exact=True,
                language_exact=True,
                tag_exact=True,
                bitrate_min=128,
                bitrate_max=320
            )
            
            mock_request.assert_called_once()
            args, kwargs = mock_request.call_args
            assert args[0] == "stations/search/byname/test"
            assert kwargs["params"]["hidebroken"] == "true"
            assert kwargs["params"]["limit"] == 10
            assert kwargs["params"]["offset"] == 5
            assert kwargs["params"]["order"] == "bitrate"
            assert kwargs["params"]["reverse"] == "true"
            assert kwargs["params"]["name_exact"] == "true"
            assert kwargs["params"]["country_exact"] == "true"
            assert kwargs["params"]["state_exact"] == "true"
            assert kwargs["params"]["language_exact"] == "true"
            assert kwargs["params"]["tag_exact"] == "true"
            assert kwargs["params"]["bitrate_min"] == 128
            assert kwargs["params"]["bitrate_max"] == 320

    @pytest.mark.asyncio
    async def test_station_found(self, radio_browser):
        """Test station method returns Station when found."""
        stations_json = """
        [
            {
                "changeuuid": "change-uuid-1",
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
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "codec": "MP3",
                "bitrate": 128,
                "hls": false,
                "lastcheckok": true,
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clickcount": 100,
                "clicktrend": 5,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": true,
                "iso_3166_2": "US-CA"
            }
        ]
        """
        
        with patch.object(radio_browser, "stations", AsyncMock(return_value=[
            Station(
                change_uuid="change-uuid-1",
                uuid="test-uuid",
                name="Test Station",
                url="http://example.com/stream",
                url_resolved="http://example.com/stream",
                homepage="http://example.com",
                favicon="http://example.com/favicon.ico",
                tags=["tag1", "tag2"],
                country_code="US",
                state="California",
                language=["English"],
                language_codes=["en"],
                votes=10,
                lastchange_time=datetime(2023, 1, 1),
                codec="MP3",
                bitrate=128,
                hls=False,
                lastcheckok=True,
                last_check_time=datetime(2023, 1, 1),
                last_check_ok_time=datetime(2023, 1, 1),
                last_local_check_time=datetime(2023, 1, 1),
                click_timestamp=datetime(2023, 1, 1),
                click_count=100,
                click_trend=5,
                ssl_error=0,
                latitude=37.7749,
                longitude=-122.4194,
                has_extended_info=True,
                iso_3166_2="US-CA"
            )
        ])):
            result = await radio_browser.station(uuid="test-uuid")
            
            assert result is not None
            assert isinstance(result, Station)
            assert result.uuid == "test-uuid"
            assert result.name == "Test Station"

    @pytest.mark.asyncio
    async def test_station_not_found(self, radio_browser):
        """Test station method returns None when not found."""
        with patch.object(radio_browser, "stations", AsyncMock(return_value=[])):
            result = await radio_browser.station(uuid="nonexistent-uuid")
            
            assert result is None

    @pytest.mark.asyncio
    async def test_stations(self, radio_browser):
        """Test stations method returns list of Station objects."""
        stations_json = """
        [
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
                "language": "English",
                "languagecodes": "en",
                "votes": 10,
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "codec": "MP3",
                "bitrate": 128,
                "hls": false,
                "lastcheckok": true,
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clickcount": 100,
                "clicktrend": 5,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": true,
                "iso_3166_2": "US-CA"
            }
        ]
        """
        
        with patch.object(radio_browser, "_request", AsyncMock(return_value=stations_json)):
            result = await radio_browser.stations()
            
            assert len(result) == 1
            assert isinstance(result[0], Station)
            assert result[0].uuid == "station-uuid-1"
            assert result[0].name == "Test Station 1"

    @pytest.mark.asyncio
    async def test_stations_with_filter(self, radio_browser):
        """Test stations method with filter parameters."""
        with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")) as mock_request:
            await radio_browser.stations(
                filter_by=FilterBy.COUNTRY,
                filter_term="US",
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.BITRATE,
                reverse=True
            )
            
            mock_request.assert_called_once()
            args, kwargs = mock_request.call_args
            assert args[0] == "stations/bycountry/US"
            assert kwargs["params"] == {
                "hidebroken": "true",
                "limit": 10,
                "offset": 5,
                "order": "bitrate",
                "reverse": "true"
            }

    @pytest.mark.asyncio
    async def test_tags(self, radio_browser):
        """Test tags method returns list of Tag objects."""
        tags_json = """
        [
            {
                "name": "rock",
                "stationcount": "500"
            },
            {
                "name": "pop",
                "stationcount": "300"
            }
        ]
        """
        
        with patch.object(radio_browser, "_request", AsyncMock(return_value=tags_json)):
            result = await radio_browser.tags()
            
            assert len(result) == 2
            assert isinstance(result[0], Tag)
            assert result[0].name == "rock"
            assert result[0].station_count == "500"
            
            assert isinstance(result[1], Tag)
            assert result[1].name == "pop"
            assert result[1].station_count == "300"

    @pytest.mark.asyncio
    async def test_tags_with_parameters(self, radio_browser):
        """Test tags method with parameters."""
        with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")) as mock_request:
            await radio_browser.tags(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.STATION_COUNT,
                reverse=True
            )
            
            mock_request.assert_called_once()
            _, kwargs = mock_request.call_args
            assert kwargs["params"] == {
                "hidebroken": "true",
                "limit": 10,
                "offset": 5,
                "order": "stationcount",
                "reverse": "true"
            }

    @pytest.mark.asyncio
    async def test_close(self, radio_browser):
        """Test close method closes session."""
        radio_browser.session = AsyncMock()
        radio_browser._close_session = True
        
        await radio_browser.close()
        
        radio_browser.session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_session(self, radio_browser):
        """Test close method does nothing when no session exists."""
        radio_browser.session = None
        
        await radio_browser.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_not_owned(self, radio_browser):
        """Test close method does not close session when not owned."""
        radio_browser.session = AsyncMock()
        radio_browser._close_session = False
        
        await radio_browser.close()
        
        radio_browser.session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_aenter(self, radio_browser):
        """Test __aenter__ returns self."""
        result = await radio_browser.__aenter__()
        
        assert result is radio_browser

    @pytest.mark.asyncio
    async def test_aexit(self, radio_browser):
        """Test __aexit__ calls close."""
        with patch.object(radio_browser, "close", AsyncMock()) as mock_close:
            await radio_browser.__aexit__(None, None, None)
            
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, radio_browser):
        """Test RadioBrowser can be used as a context manager."""
        with patch.object(radio_browser, "close", AsyncMock()) as mock_close:
            async with radio_browser as rb:
                assert rb is radio_browser
            
            mock_close.assert_called_once()