import asyncio
import socket
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
    """Return a mocked aiohttp ClientSession."""
    session = AsyncMock()
    response = AsyncMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    response.text = AsyncMock(return_value='{"test": "data"}')
    session.request.return_value = response
    return session


@pytest.fixture
def mock_dns_resolver():
    """Return a mocked DNSResolver."""
    resolver = AsyncMock()
    result = [MagicMock(host="api.radio-browser.info")]
    resolver.query.return_value = result
    return resolver


class TestRadioBrowser:
    """Tests for the RadioBrowser class."""

    @pytest.mark.asyncio
    async def test_request_dns_resolution(self, radio_browser, mock_session):
        """Test that _request resolves DNS when host is None."""
        radio_browser.session = mock_session
        radio_browser._host = None

        with patch("radios.radio_browser.DNSResolver") as mock_resolver_class:
            mock_resolver = AsyncMock()
            mock_result = [MagicMock(host="api.radio-browser.info")]
            mock_resolver.query.return_value = mock_result
            mock_resolver_class.return_value = mock_resolver

            await radio_browser._request("test")

            mock_resolver.query.assert_called_once_with(
                "_api._tcp.radio-browser.info", "SRV"
            )
            assert radio_browser._host == "api.radio-browser.info"

    @pytest.mark.asyncio
    async def test_request_creates_session_if_none(self, radio_browser):
        """Test that _request creates a session if none exists."""
        radio_browser._host = "api.radio-browser.info"
        radio_browser.session = None

        with patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.text = AsyncMock(return_value='{"test": "data"}')
            mock_session.request.return_value = mock_response
            mock_session_class.return_value = mock_session

            await radio_browser._request("test")

            mock_session_class.assert_called_once()
            assert radio_browser.session is not None
            assert radio_browser._close_session is True

    @pytest.mark.asyncio
    async def test_request_timeout(self, radio_browser, mock_session):
        """Test that _request raises RadioBrowserConnectionTimeoutError on timeout."""
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        mock_session.request.side_effect = asyncio.TimeoutError()

        with pytest.raises(RadioBrowserConnectionTimeoutError):
            await radio_browser._request("test")

        assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_connection_error(self, radio_browser, mock_session):
        """Test that _request raises RadioBrowserConnectionError on connection error."""
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        mock_session.request.side_effect = aiohttp.ClientError()

        with pytest.raises(RadioBrowserConnectionError):
            await radio_browser._request("test")

        assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_socket_error(self, radio_browser, mock_session):
        """Test that _request raises RadioBrowserConnectionError on socket error."""
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        mock_session.request.side_effect = socket.gaierror()

        with pytest.raises(RadioBrowserConnectionError):
            await radio_browser._request("test")

        assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_invalid_content_type(self, radio_browser, mock_session):
        """Test that _request raises RadioBrowserError on invalid content type."""
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.text = AsyncMock(return_value="Not JSON")
        mock_session.request.return_value = mock_response

        with pytest.raises(RadioBrowserError):
            await radio_browser._request("test")

    @pytest.mark.asyncio
    async def test_request_boolean_params(self, radio_browser, mock_session):
        """Test that _request converts boolean params to lowercase strings."""
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"

        await radio_browser._request("test", params={"bool_param": True})

        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args[1]
        assert call_args["params"]["bool_param"] == "true"

    @pytest.mark.asyncio
    async def test_stats(self, radio_browser):
        """Test the stats method."""
        stats_data = {
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
        stats_json = orjson.dumps(stats_data)

        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=stats_json)
        ):
            result = await radio_browser.stats()

            radio_browser._request.assert_called_once_with("stats")
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
        """Test the station_click method."""
        with patch.object(radio_browser, "_request", AsyncMock()):
            await radio_browser.station_click(uuid="test-uuid")

            radio_browser._request.assert_called_once_with("url/test-uuid")

    @pytest.mark.asyncio
    async def test_countries(self, radio_browser):
        """Test the countries method."""
        countries_data = [
            {"name": "US", "stationcount": "100"},
            {"name": "GB", "stationcount": "50"},
            {"name": "XK", "stationcount": "10"},  # Kosovo special case
        ]
        countries_json = orjson.dumps(countries_data)

        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=countries_json)
        ):
            result = await radio_browser.countries()

            radio_browser._request.assert_called_once_with(
                "countrycodes",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                },
            )
            assert isinstance(result, list)
            assert len(result) == 3
            assert all(isinstance(country, Country) for country in result)
            
            # Check Kosovo special case
            kosovo = next(c for c in result if c.code == "XK")
            assert kosovo.name == "Kosovo"
            
            # Check country name resolution
            us = next(c for c in result if c.code == "US")
            assert us.name == "United States"

    @pytest.mark.asyncio
    async def test_countries_with_parameters(self, radio_browser):
        """Test the countries method with custom parameters."""
        countries_data = [{"name": "US", "stationcount": "100"}]
        countries_json = orjson.dumps(countries_data)

        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=countries_json)
        ):
            result = await radio_browser.countries(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.STATION_COUNT,
                reverse=True,
            )

            radio_browser._request.assert_called_once_with(
                "countrycodes",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "stationcount",
                    "reverse": True,
                },
            )
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_languages(self, radio_browser):
        """Test the languages method."""
        languages_data = [
            {"name": "english", "stationcount": "100", "iso_639": "en"},
            {"name": "spanish", "stationcount": "50", "iso_639": "es"},
        ]
        languages_json = orjson.dumps(languages_data)

        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=languages_json)
        ):
            result = await radio_browser.languages()

            radio_browser._request.assert_called_once_with(
                "languages",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                },
            )
            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(language, Language) for language in result)
            
            # Check title case conversion
            english = next(l for l in result if l.code == "en")
            assert english.name == "English"

    @pytest.mark.asyncio
    async def test_languages_with_parameters(self, radio_browser):
        """Test the languages method with custom parameters."""
        languages_data = [{"name": "english", "stationcount": "100", "iso_639": "en"}]
        languages_json = orjson.dumps(languages_data)

        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=languages_json)
        ):
            result = await radio_browser.languages(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.STATION_COUNT,
                reverse=True,
            )

            radio_browser._request.assert_called_once_with(
                "languages",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "stationcount",
                    "reverse": True,
                },
            )
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_search(self, radio_browser):
        """Test the search method."""
        stations_data = [
            {
                "bitrate": 128,
                "changeuuid": "change-uuid",
                "clickcount": 100,
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clicktrend": 1,
                "codec": "MP3",
                "countrycode": "US",
                "favicon": "https://example.com/favicon.ico",
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "hls": False,
                "homepage": "https://example.com",
                "iso_3166_2": "US-CA",
                "language": "English",
                "languagecodes": "en",
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckok": True,
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "name": "Test Station",
                "ssl_error": 0,
                "state": "California",
                "stationuuid": "station-uuid",
                "tags": "rock,pop",
                "url_resolved": "https://example.com/stream",
                "url": "https://example.com/stream",
                "votes": 50,
            }
        ]
        stations_json = orjson.dumps(stations_data)

        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=stations_json)
        ):
            result = await radio_browser.search(name="Test")

            radio_browser._request.assert_called_once_with(
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
                },
            )
            assert isinstance(result, list)
            assert len(result) == 1
            assert all(isinstance(station, Station) for station in result)
            assert result[0].name == "Test Station"

    @pytest.mark.asyncio
    async def test_search_with_filter(self, radio_browser):
        """Test the search method with filter parameters."""
        stations_data = [
            {
                "bitrate": 128,
                "changeuuid": "change-uuid",
                "clickcount": 100,
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clicktrend": 1,
                "codec": "MP3",
                "countrycode": "US",
                "favicon": "https://example.com/favicon.ico",
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "hls": False,
                "homepage": "https://example.com",
                "iso_3166_2": "US-CA",
                "language": "English",
                "languagecodes": "en",
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckok": True,
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "name": "Test Station",
                "ssl_error": 0,
                "state": "California",
                "stationuuid": "station-uuid",
                "tags": "rock,pop",
                "url_resolved": "https://example.com/stream",
                "url": "https://example.com/stream",
                "votes": 50,
            }
        ]
        stations_json = orjson.dumps(stations_data)

        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=stations_json)
        ):
            result = await radio_browser.search(
                filter_by=FilterBy.COUNTRY,
                filter_term="US",
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
                bitrate_max=320,
            )

            radio_browser._request.assert_called_once_with(
                "stations/search/bycountry/US",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "bitrate",
                    "reverse": True,
                    "name": None,
                    "name_exact": True,
                    "country": "",
                    "country_exact": True,
                    "state_exact": True,
                    "language_exact": True,
                    "tag_exact": True,
                    "bitrate_min": 128,
                    "bitrate_max": 320,
                },
            )
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_station(self, radio_browser):
        """Test the station method."""
        stations_data = [
            {
                "bitrate": 128,
                "changeuuid": "change-uuid",
                "clickcount": 100,
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clicktrend": 1,
                "codec": "MP3",
                "countrycode": "US",
                "favicon": "https://example.com/favicon.ico",
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "hls": False,
                "homepage": "https://example.com",
                "iso_3166_2": "US-CA",
                "language": "English",
                "languagecodes": "en",
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckok": True,
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "name": "Test Station",
                "ssl_error": 0,
                "state": "California",
                "stationuuid": "station-uuid",
                "tags": "rock,pop",
                "url_resolved": "https://example.com/stream",
                "url": "https://example.com/stream",
                "votes": 50,
            }
        ]
        stations_json = orjson.dumps(stations_data)

        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=stations_json)
        ):
            result = await radio_browser.station(uuid="station-uuid")

            assert isinstance(result, Station)
            assert result.uuid == "station-uuid"
            assert result.name == "Test Station"

    @pytest.mark.asyncio
    async def test_station_not_found(self, radio_browser):
        """Test the station method when station is not found."""
        with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")):
            result = await radio_browser.station(uuid="non-existent-uuid")

            assert result is None

    @pytest.mark.asyncio
    async def test_stations(self, radio_browser):
        """Test the stations method."""
        stations_data = [
            {
                "bitrate": 128,
                "changeuuid": "change-uuid",
                "clickcount": 100,
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clicktrend": 1,
                "codec": "MP3",
                "countrycode": "US",
                "favicon": "https://example.com/favicon.ico",
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "hls": False,
                "homepage": "https://example.com",
                "iso_3166_2": "US-CA",
                "language": "English",
                "languagecodes": "en",
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckok": True,
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "name": "Test Station",
                "ssl_error": 0,
                "state": "California",
                "stationuuid": "station-uuid",
                "tags": "rock,pop",
                "url_resolved": "https://example.com/stream",
                "url": "https://example.com/stream",
                "votes": 50,
            }
        ]
        stations_json = orjson.dumps(stations_data)

        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=stations_json)
        ):
            result = await radio_browser.stations()

            radio_browser._request.assert_called_once_with(
                "stations",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                },
            )
            assert isinstance(result, list)
            assert len(result) == 1
            assert all(isinstance(station, Station) for station in result)
            assert result[0].name == "Test Station"

    @pytest.mark.asyncio
    async def test_stations_with_filter(self, radio_browser):
        """Test the stations method with filter parameters."""
        stations_data = [
            {
                "bitrate": 128,
                "changeuuid": "change-uuid",
                "clickcount": 100,
                "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
                "clicktrend": 1,
                "codec": "MP3",
                "countrycode": "US",
                "favicon": "https://example.com/favicon.ico",
                "geo_lat": 37.7749,
                "geo_long": -122.4194,
                "has_extended_info": True,
                "hls": False,
                "homepage": "https://example.com",
                "iso_3166_2": "US-CA",
                "language": "English",
                "languagecodes": "en",
                "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
                "lastcheckok": True,
                "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
                "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
                "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
                "name": "Test Station",
                "ssl_error": 0,
                "state": "California",
                "stationuuid": "station-uuid",
                "tags": "rock,pop",
                "url_resolved": "https://example.com/stream",
                "url": "https://example.com/stream",
                "votes": 50,
            }
        ]
        stations_json = orjson.dumps(stations_data)

        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=stations_json)
        ):
            result = await radio_browser.stations(
                filter_by=FilterBy.COUNTRY,
                filter_term="US",
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.BITRATE,
                reverse=True,
            )

            radio_browser._request.assert_called_once_with(
                "stations/bycountry/US",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "bitrate",
                    "reverse": True,
                },
            )
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_tags(self, radio_browser):
        """Test the tags method."""
        tags_data = [
            {"name": "rock", "stationcount": "100"},
            {"name": "pop", "stationcount": "50"},
        ]
        tags_json = orjson.dumps(tags_data)

        with patch.object(radio_browser, "_request", AsyncMock(return_value=tags_json)):
            result = await radio_browser.tags()

            radio_browser._request.assert_called_once_with(
                "tags",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                },
            )
            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(tag, Tag) for tag in result)
            assert result[0].name == "rock"
            assert result[0].station_count == "100"
            assert result[1].name == "pop"
            assert result[1].station_count == "50"

    @pytest.mark.asyncio
    async def test_tags_with_parameters(self, radio_browser):
        """Test the tags method with custom parameters."""
        tags_data = [{"name": "rock", "stationcount": "100"}]
        tags_json = orjson.dumps(tags_data)

        with patch.object(radio_browser, "_request", AsyncMock(return_value=tags_json)):
            result = await radio_browser.tags(
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.STATION_COUNT,
                reverse=True,
            )

            radio_browser._request.assert_called_once_with(
                "tags",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "stationcount",
                    "reverse": True,
                },
            )
            assert len(result) == 1
            assert result[0].name == "rock"
            assert result[0].station_count == "100"

    @pytest.mark.asyncio
    async def test_close(self, radio_browser, mock_session):
        """Test the close method."""
        radio_browser.session = mock_session
        radio_browser._close_session = True

        await radio_browser.close()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_session(self, radio_browser):
        """Test the close method when no session exists."""
        radio_browser.session = None
        radio_browser._close_session = True

        await radio_browser.close()  # Should not raise an exception

    @pytest.mark.asyncio
    async def test_close_not_owned(self, radio_browser, mock_session):
        """Test the close method when session is not owned."""
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
        with patch.object(radio_browser, "close", AsyncMock()) as mock_close:
            await radio_browser.__aexit__(None, None, None)
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, radio_browser):
        """Test using RadioBrowser as a context manager."""
        with patch.object(radio_browser, "close", AsyncMock()) as mock_close:
            async with radio_browser as rb:
                assert rb is radio_browser
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_backoff_retry(self, radio_browser, mock_session):
        """Test that _request retries on connection error."""
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        
        # First call raises error, second succeeds
        mock_session.request.side_effect = [
            aiohttp.ClientError(),
            AsyncMock(
                status=200,
                headers={"Content-Type": "application/json"},
                text=AsyncMock(return_value='{"test": "data"}')
            )
        ]
        
        # Patch backoff to make tests faster
        with patch("radios.radio_browser.backoff.expo", return_value=[0.1]):
            result = await radio_browser._request("test")
            
            assert result == '{"test": "data"}'
            assert mock_session.request.call_count == 2

    @pytest.mark.asyncio
    async def test_search_with_all_parameters(self, radio_browser):
        """Test the search method with all parameters."""
        stations_data = []
        stations_json = orjson.dumps(stations_data)

        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=stations_json)
        ):
            result = await radio_browser.search(
                filter_by=FilterBy.NAME,
                filter_term="Test",
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
                bitrate_min=128,
                bitrate_max=320,
            )

            radio_browser._request.assert_called_once_with(
                "stations/search/byname/Test",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "bitrate",
                    "reverse": True,
                    "name": "Test",
                    "name_exact": True,
                    "country": "US",
                    "country_exact": True,
                    "state_exact": True,
                    "language_exact": True,
                    "tag_exact": True,
                    "bitrate_min": 128,
                    "bitrate_max": 320,
                },
            )
            assert result == []  # TODO: Fix this assertion - should be an empty list