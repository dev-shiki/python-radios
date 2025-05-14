import asyncio
import socket
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
def radio_browser():
    """Return a RadioBrowser instance."""
    return RadioBrowser(user_agent="Test/1.0")


@pytest.fixture
def mock_dns_resolver():
    """Return a mocked DNS resolver."""
    resolver = MagicMock(spec=DNSResolver)
    result = MagicMock()
    result.host = "api.radio-browser.info"
    resolver.query.return_value = asyncio.Future()
    resolver.query.return_value.set_result([result])
    return resolver


@pytest.fixture
def mock_session():
    """Return a mocked aiohttp ClientSession."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    return session


@pytest.fixture
def mock_response():
    """Return a mocked aiohttp ClientResponse."""
    response = AsyncMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    response.text = AsyncMock(return_value='{"key": "value"}')
    return response


class TestRadioBrowser:
    """Tests for the RadioBrowser class."""

    @pytest.mark.asyncio
    async def test_request_dns_resolution(self, radio_browser, mock_dns_resolver):
        """Test that the _request method resolves DNS."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            with patch.object(radio_browser, "session", AsyncMock()) as mock_session:
                mock_response = AsyncMock()
                mock_response.headers = {"Content-Type": "application/json"}
                mock_response.text.return_value = "{}"
                mock_session.request.return_value = mock_response

                await radio_browser._request("test")

                mock_dns_resolver.query.assert_called_once_with(
                    "_api._tcp.radio-browser.info", "SRV"
                )
                assert radio_browser._host == "api.radio-browser.info"

    @pytest.mark.asyncio
    async def test_request_creates_session_if_none(self, radio_browser, mock_dns_resolver):
        """Test that _request creates a session if none exists."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            with patch("radios.radio_browser.aiohttp.ClientSession") as mock_client_session:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.headers = {"Content-Type": "application/json"}
                mock_response.text.return_value = "{}"
                mock_session.request.return_value = mock_response
                mock_client_session.return_value = mock_session

                assert radio_browser.session is None
                await radio_browser._request("test")
                assert radio_browser.session is not None
                assert radio_browser._close_session is True

    @pytest.mark.asyncio
    async def test_request_uses_existing_session(self, radio_browser, mock_dns_resolver):
        """Test that _request uses an existing session if available."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.text.return_value = "{}"
            mock_session.request.return_value = mock_response
            radio_browser.session = mock_session
            radio_browser._close_session = False

            await radio_browser._request("test")

            mock_session.request.assert_called_once()
            assert radio_browser._close_session is False

    @pytest.mark.asyncio
    async def test_request_timeout(self, radio_browser, mock_dns_resolver):
        """Test that _request raises RadioBrowserConnectionTimeoutError on timeout."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            mock_session = AsyncMock()
            mock_session.request.side_effect = asyncio.TimeoutError()
            radio_browser.session = mock_session

            with pytest.raises(RadioBrowserConnectionTimeoutError):
                await radio_browser._request("test")

            assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_client_error(self, radio_browser, mock_dns_resolver):
        """Test that _request raises RadioBrowserConnectionError on client error."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            mock_session = AsyncMock()
            mock_session.request.side_effect = aiohttp.ClientError()
            radio_browser.session = mock_session

            with pytest.raises(RadioBrowserConnectionError):
                await radio_browser._request("test")

            assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_socket_error(self, radio_browser, mock_dns_resolver):
        """Test that _request raises RadioBrowserConnectionError on socket error."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            mock_session = AsyncMock()
            mock_session.request.side_effect = socket.gaierror()
            radio_browser.session = mock_session

            with pytest.raises(RadioBrowserConnectionError):
                await radio_browser._request("test")

            assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_invalid_content_type(self, radio_browser, mock_dns_resolver):
        """Test that _request raises RadioBrowserError on invalid content type."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"Content-Type": "text/plain"}
            mock_response.text.return_value = "Not JSON"
            mock_session.request.return_value = mock_response
            radio_browser.session = mock_session

            with pytest.raises(RadioBrowserError):
                await radio_browser._request("test")

    @pytest.mark.asyncio
    async def test_request_boolean_params(self, radio_browser, mock_dns_resolver):
        """Test that _request converts boolean params to lowercase strings."""
        with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.text.return_value = "{}"
            mock_session.request.return_value = mock_response
            radio_browser.session = mock_session

            await radio_browser._request("test", params={"bool_param": True})

            mock_session.request.assert_called_once()
            call_args = mock_session.request.call_args
            assert call_args[1]["params"]["bool_param"] == "true"

    @pytest.mark.asyncio
    async def test_stats(self, radio_browser):
        """Test the stats method."""
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
        
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(stats_data)
            
            result = await radio_browser.stats()
            
            mock_request.assert_called_once_with("stats")
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

    @pytest.mark.asyncio
    async def test_station_click(self, radio_browser):
        """Test the station_click method."""
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = "{}"
            
            await radio_browser.station_click(uuid="test-uuid")
            
            mock_request.assert_called_once_with("url/test-uuid")

    @pytest.mark.asyncio
    async def test_countries(self, radio_browser):
        """Test the countries method with default parameters."""
        countries_data = [
            {
                "name": "US",
                "stationcount": "100"
            },
            {
                "name": "GB",
                "stationcount": "50"
            },
            {
                "name": "XK",  # Kosovo special case
                "stationcount": "10"
            }
        ]
        
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(countries_data)
            
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
            
            assert len(result) == 3
            assert isinstance(result[0], Country)
            
            # Check country name resolution
            us_country = next(c for c in result if c.code == "US")
            gb_country = next(c for c in result if c.code == "GB")
            kosovo_country = next(c for c in result if c.code == "XK")
            
            assert us_country.name == "United States"
            assert gb_country.name == "United Kingdom"
            assert kosovo_country.name == "Kosovo"

    @pytest.mark.asyncio
    async def test_countries_with_custom_parameters(self, radio_browser):
        """Test the countries method with custom parameters."""
        countries_data = [
            {
                "name": "US",
                "stationcount": "100"
            }
        ]
        
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(countries_data)
            
            result = await radio_browser.countries(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.STATION_COUNT,
                reverse=True
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
            
            assert len(result) == 1
            assert isinstance(result[0], Country)

    @pytest.mark.asyncio
    async def test_languages(self, radio_browser):
        """Test the languages method with default parameters."""
        languages_data = [
            {
                "name": "english",
                "iso_639": "en",
                "stationcount": "100"
            },
            {
                "name": "spanish",
                "iso_639": "es",
                "stationcount": "50"
            }
        ]
        
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(languages_data)
            
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
            
            assert len(result) == 2
            assert isinstance(result[0], Language)
            
            # Check language name capitalization
            assert result[0].name == "English"
            assert result[1].name == "Spanish"
            assert result[0].code == "en"
            assert result[1].code == "es"
            assert result[0].station_count == "100"
            assert result[1].station_count == "50"

    @pytest.mark.asyncio
    async def test_languages_with_custom_parameters(self, radio_browser):
        """Test the languages method with custom parameters."""
        languages_data = [
            {
                "name": "english",
                "iso_639": "en",
                "stationcount": "100"
            }
        ]
        
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(languages_data)
            
            result = await radio_browser.languages(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.STATION_COUNT,
                reverse=True
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
            
            assert len(result) == 1
            assert isinstance(result[0], Language)
            assert result[0].name == "English"

    @pytest.mark.asyncio
    async def test_search_basic(self, radio_browser):
        """Test the search method with basic parameters."""
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
                "language": "English",
                "languagecodes": "en",
                "votes": 10,
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "codec": "MP3",
                "bitrate": 128,
                "hls": False,
                "lastcheckok": True,
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clickcount": 100,
                "clicktrend": 1,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "iso_3166_2": "US-CA"
            }
        ]
        
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(stations_data)
            
            result = await radio_browser.search(name="Test")
            
            mock_request.assert_called_once_with(
                "stations/search",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                    "name": "Test",
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
            
            assert len(result) == 1
            assert isinstance(result[0], Station)
            assert result[0].name == "Test Station 1"
            assert result[0].uuid == "station-uuid-1"

    @pytest.mark.asyncio
    async def test_search_with_filter(self, radio_browser):
        """Test the search method with filter parameters."""
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
                "language": "English",
                "languagecodes": "en",
                "votes": 10,
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "codec": "MP3",
                "bitrate": 128,
                "hls": False,
                "lastcheckok": True,
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clickcount": 100,
                "clicktrend": 1,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "iso_3166_2": "US-CA"
            }
        ]
        
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(stations_data)
            
            result = await radio_browser.search(
                filter_by=FilterBy.TAG,
                filter_term="rock",
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.VOTES,
                reverse=True,
                name_exact=True,
                country="US",
                country_exact=True,
                state_exact=True,
                language_exact=True,
                tag_exact=True,
                bitrate_min=128,
                bitrate_max=320
            )
            
            mock_request.assert_called_once_with(
                "stations/search/bytag/rock",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "votes",
                    "reverse": True,
                    "name": None,
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
            
            assert len(result) == 1
            assert isinstance(result[0], Station)

    @pytest.mark.asyncio
    async def test_station_found(self, radio_browser):
        """Test the station method when a station is found."""
        stations_data = [
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
                "hls": False,
                "lastcheckok": True,
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clickcount": 100,
                "clicktrend": 1,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "iso_3166_2": "US-CA"
            }
        ]
        
        with patch.object(radio_browser, "stations") as mock_stations:
            mock_stations.return_value = [Station.from_dict(stations_data[0])]
            
            result = await radio_browser.station(uuid="test-uuid")
            
            mock_stations.assert_called_once_with(
                filter_by=FilterBy.UUID,
                filter_term="test-uuid",
                limit=1,
            )
            
            assert isinstance(result, Station)
            assert result.uuid == "test-uuid"
            assert result.name == "Test Station"

    @pytest.mark.asyncio
    async def test_station_not_found(self, radio_browser):
        """Test the station method when a station is not found."""
        with patch.object(radio_browser, "stations") as mock_stations:
            mock_stations.return_value = []
            
            result = await radio_browser.station(uuid="nonexistent-uuid")
            
            mock_stations.assert_called_once_with(
                filter_by=FilterBy.UUID,
                filter_term="nonexistent-uuid",
                limit=1,
            )
            
            assert result is None

    @pytest.mark.asyncio
    async def test_stations_basic(self, radio_browser):
        """Test the stations method with basic parameters."""
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
                "language": "English",
                "languagecodes": "en",
                "votes": 10,
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "codec": "MP3",
                "bitrate": 128,
                "hls": False,
                "lastcheckok": True,
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clickcount": 100,
                "clicktrend": 1,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "iso_3166_2": "US-CA"
            }
        ]
        
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(stations_data)
            
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
            
            assert len(result) == 1
            assert isinstance(result[0], Station)
            assert result[0].name == "Test Station 1"
            assert result[0].uuid == "station-uuid-1"

    @pytest.mark.asyncio
    async def test_stations_with_filter(self, radio_browser):
        """Test the stations method with filter parameters."""
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
                "language": "English",
                "languagecodes": "en",
                "votes": 10,
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "codec": "MP3",
                "bitrate": 128,
                "hls": False,
                "lastcheckok": True,
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clickcount": 100,
                "clicktrend": 1,
                "ssl_error": 0,
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "iso_3166_2": "US-CA"
            }
        ]
        
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(stations_data)
            
            result = await radio_browser.stations(
                filter_by=FilterBy.COUNTRY,
                filter_term="US",
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.VOTES,
                reverse=True
            )
            
            mock_request.assert_called_once_with(
                "stations/bycountry/US",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "votes",
                    "reverse": True,
                }
            )
            
            assert len(result) == 1
            assert isinstance(result[0], Station)

    @pytest.mark.asyncio
    async def test_tags(self, radio_browser):
        """Test the tags method with default parameters."""
        tags_data = [
            {
                "name": "rock",
                "stationcount": "100"
            },
            {
                "name": "pop",
                "stationcount": "50"
            }
        ]
        
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(tags_data)
            
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
            
            assert len(result) == 2
            assert isinstance(result[0], Tag)
            assert result[0].name == "rock"
            assert result[0].station_count == "100"
            assert result[1].name == "pop"
            assert result[1].station_count == "50"

    @pytest.mark.asyncio
    async def test_tags_with_custom_parameters(self, radio_browser):
        """Test the tags method with custom parameters."""
        tags_data = [
            {
                "name": "rock",
                "stationcount": "100"
            }
        ]
        
        with patch.object(radio_browser, "_request") as mock_request:
            mock_request.return_value = orjson.dumps(tags_data)
            
            result = await radio_browser.tags(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.STATION_COUNT,
                reverse=True
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
            
            assert len(result) == 1
            assert isinstance(result[0], Tag)
            assert result[0].name == "rock"
            assert result[0].station_count == "100"  # TODO: Fix this assertion

    @pytest.mark.asyncio
    async def test_close(self, radio_browser):
        """Test the close method."""
        mock_session = AsyncMock()
        radio_browser.session = mock_session
        radio_browser._close_session = True
        
        await radio_browser.close()
        
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_session(self, radio_browser):
        """Test the close method when no session exists."""
        radio_browser.session = None
        
        await radio_browser.close()  # Should not raise an exception

    @pytest.mark.asyncio
    async def test_close_not_owned_session(self, radio_browser):
        """Test the close method when session is not owned."""
        mock_session = AsyncMock()
        radio_browser.session = mock_session
        radio_browser._close_session = False
        
        await radio_browser.close()
        
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_aenter(self, radio_browser):
        """Test the __aenter__ method."""
        result = await radio_browser.__aenter__()
        assert result is radio_browser

    @pytest.mark.asyncio
    async def test_aexit(self, radio_browser):
        """Test the __aexit__ method."""
        with patch.object(radio_browser, "close") as mock_close:
            await radio_browser.__aexit__(None, None, None)
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, radio_browser):
        """Test using RadioBrowser as an async context manager."""
        with patch.object(radio_browser, "close") as mock_close:
            async with radio_browser as rb:
                assert rb is radio_browser
            
            mock_close.assert_called_once()