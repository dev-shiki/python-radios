import asyncio
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
    response = AsyncMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    response.text = AsyncMock(return_value='{"test": "data"}')
    session.request.return_value = response
    return session


@pytest.fixture
def mock_dns_resolver():
    """Return a mock DNSResolver."""
    resolver = AsyncMock()
    result = [MagicMock(host="api.radio-browser.info")]
    resolver.query.return_value = result
    return resolver


class TestRadioBrowser:
    """Tests for the RadioBrowser class."""

    @pytest.mark.asyncio
    async def test_request_dns_resolution(self, radio_browser, mock_session):
        """Test DNS resolution in _request method."""
        with patch("radios.radio_browser.DNSResolver") as mock_resolver_class:
            mock_resolver = AsyncMock()
            mock_result = [MagicMock(host="api.radio-browser.info")]
            mock_resolver.query.return_value = mock_result
            mock_resolver_class.return_value = mock_resolver

            radio_browser.session = mock_session
            await radio_browser._request("test")

            mock_resolver.query.assert_called_once_with(
                "_api._tcp.radio-browser.info", "SRV"
            )
            assert radio_browser._host == "api.radio-browser.info"

    @pytest.mark.asyncio
    async def test_request_uses_cached_host(self, radio_browser, mock_session):
        """Test _request uses cached host."""
        radio_browser._host = "cached.radio-browser.info"
        radio_browser.session = mock_session

        await radio_browser._request("test")

        mock_session.request.assert_called_once()
        # Verify the URL contains the cached host
        url = mock_session.request.call_args[0][1]
        assert "cached.radio-browser.info" in str(url)

    @pytest.mark.asyncio
    async def test_request_creates_session_if_none(self, radio_browser):
        """Test _request creates a session if none exists."""
        with patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            response = AsyncMock()
            response.status = 200
            response.headers = {"Content-Type": "application/json"}
            response.text = AsyncMock(return_value='{"test": "data"}')
            mock_session.request.return_value = response
            mock_session_class.return_value = mock_session

            with patch("radios.radio_browser.DNSResolver") as mock_resolver_class:
                mock_resolver = AsyncMock()
                mock_result = [MagicMock(host="api.radio-browser.info")]
                mock_resolver.query.return_value = mock_result
                mock_resolver_class.return_value = mock_resolver

                await radio_browser._request("test")

                mock_session_class.assert_called_once()
                assert radio_browser.session is not None
                assert radio_browser._close_session is True

    @pytest.mark.asyncio
    async def test_request_timeout(self, radio_browser, mock_session):
        """Test _request handles timeout."""
        radio_browser._host = "api.radio-browser.info"
        radio_browser.session = mock_session
        mock_session.request.side_effect = asyncio.TimeoutError()

        with pytest.raises(RadioBrowserConnectionTimeoutError):
            await radio_browser._request("test")

        assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_connection_error(self, radio_browser, mock_session):
        """Test _request handles connection error."""
        radio_browser._host = "api.radio-browser.info"
        radio_browser.session = mock_session
        mock_session.request.side_effect = aiohttp.ClientError()

        with pytest.raises(RadioBrowserConnectionError):
            await radio_browser._request("test")

        assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_socket_error(self, radio_browser, mock_session):
        """Test _request handles socket error."""
        radio_browser._host = "api.radio-browser.info"
        radio_browser.session = mock_session
        mock_session.request.side_effect = socket.gaierror()

        with pytest.raises(RadioBrowserConnectionError):
            await radio_browser._request("test")

        assert radio_browser._host is None

    @pytest.mark.asyncio
    async def test_request_invalid_content_type(self, radio_browser, mock_session):
        """Test _request handles invalid content type."""
        radio_browser._host = "api.radio-browser.info"
        radio_browser.session = mock_session
        response = AsyncMock()
        response.status = 200
        response.headers = {"Content-Type": "text/html"}
        response.text = AsyncMock(return_value="<html>Not JSON</html>")
        mock_session.request.return_value = response

        with pytest.raises(RadioBrowserError):
            await radio_browser._request("test")

    @pytest.mark.asyncio
    async def test_request_boolean_params(self, radio_browser, mock_session):
        """Test _request converts boolean params to lowercase strings."""
        radio_browser._host = "api.radio-browser.info"
        radio_browser.session = mock_session

        await radio_browser._request("test", params={"bool_param": True, "another_bool": False})

        # Check that the boolean params were converted to lowercase strings
        call_args = mock_session.request.call_args
        assert "bool_param=true" in str(call_args) or call_args[1]["params"]["bool_param"] == "true"
        assert "another_bool=false" in str(call_args) or call_args[1]["params"]["another_bool"] == "false"

    @pytest.mark.asyncio
    async def test_stats(self, radio_browser):
        """Test stats method."""
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
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=stats_json)
        ) as mock_request:
            result = await radio_browser.stats()

            mock_request.assert_called_once_with("stats")
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
        """Test station_click method."""
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value='{"status": "ok"}')
        ) as mock_request:
            await radio_browser.station_click(uuid="test-uuid")

            mock_request.assert_called_once_with("url/test-uuid")

    @pytest.mark.asyncio
    async def test_countries(self, radio_browser):
        """Test countries method."""
        countries_json = """
        [
            {
                "name": "US",
                "stationcount": "100"
            },
            {
                "name": "XK",
                "stationcount": "50"
            },
            {
                "name": "DE",
                "stationcount": "200"
            }
        ]
        """
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=countries_json)
        ) as mock_request:
            result = await radio_browser.countries()

            mock_request.assert_called_once_with(
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
            
            # Find countries by code
            us_country = next((c for c in result if c.code == "US"), None)
            kosovo_country = next((c for c in result if c.code == "XK"), None)
            de_country = next((c for c in result if c.code == "DE"), None)
            
            assert us_country is not None
            assert us_country.name == "United States"
            assert us_country.station_count == "100"
            
            assert kosovo_country is not None
            assert kosovo_country.name == "Kosovo"
            assert kosovo_country.station_count == "50"
            
            assert de_country is not None
            assert de_country.name == "Germany"
            assert de_country.station_count == "200"

    @pytest.mark.asyncio
    async def test_countries_with_parameters(self, radio_browser):
        """Test countries method with parameters."""
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value="[]")
        ) as mock_request:
            await radio_browser.countries(
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
                },
            )

    @pytest.mark.asyncio
    async def test_languages(self, radio_browser):
        """Test languages method."""
        languages_json = """
        [
            {
                "name": "english",
                "iso_639": "en",
                "stationcount": "500"
            },
            {
                "name": "german",
                "iso_639": "de",
                "stationcount": "300"
            },
            {
                "name": "french",
                "iso_639": "fr",
                "stationcount": "200"
            }
        ]
        """
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=languages_json)
        ) as mock_request:
            result = await radio_browser.languages()

            mock_request.assert_called_once_with(
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
            assert len(result) == 3
            assert all(isinstance(language, Language) for language in result)
            
            # Find languages by code
            en_language = next((l for l in result if l.code == "en"), None)
            de_language = next((l for l in result if l.code == "de"), None)
            fr_language = next((l for l in result if l.code == "fr"), None)
            
            assert en_language is not None
            assert en_language.name == "English"
            assert en_language.station_count == "500"
            
            assert de_language is not None
            assert de_language.name == "German"
            assert de_language.station_count == "300"
            
            assert fr_language is not None
            assert fr_language.name == "French"
            assert fr_language.station_count == "200"

    @pytest.mark.asyncio
    async def test_languages_with_parameters(self, radio_browser):
        """Test languages method with parameters."""
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value="[]")
        ) as mock_request:
            await radio_browser.languages(
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
                },
            )

    @pytest.mark.asyncio
    async def test_search(self, radio_browser):
        """Test search method."""
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
                "language": "english",
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
                "has_extended_info": false,
                "iso_3166_2": "US-CA"
            }
        ]
        """
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=stations_json)
        ) as mock_request:
            result = await radio_browser.search(
                filter_by=FilterBy.NAME,
                filter_term="test",
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.NAME,
                reverse=True,
                name="Test Station",
                name_exact=True,
                country="United States",
                country_exact=True,
                state_exact=True,
                language_exact=True,
                tag_exact=True,
                bitrate_min=64,
                bitrate_max=320,
            )

            mock_request.assert_called_once_with(
                "stations/search/byname/test",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "name",
                    "reverse": True,
                    "name": "Test Station",
                    "name_exact": True,
                    "country": "United States",
                    "country_exact": True,
                    "state_exact": True,
                    "language_exact": True,
                    "tag_exact": True,
                    "bitrate_min": 64,
                    "bitrate_max": 320,
                },
            )
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], Station)
            assert result[0].uuid == "station-uuid-1"
            assert result[0].name == "Test Station 1"
            assert result[0].url == "http://example.com/stream1"
            assert result[0].url_resolved == "http://example.com/stream1"
            assert result[0].homepage == "http://example.com"
            assert result[0].favicon == "http://example.com/favicon.ico"
            assert result[0].tags == ["tag1", "tag2"]
            assert result[0].country_code == "US"
            assert result[0].state == "California"
            assert result[0].language == ["english"]
            assert result[0].language_codes == ["en"]
            assert result[0].votes == 10
            assert result[0].lastchange_time == datetime(2023, 1, 1, 0, 0, 0)
            assert result[0].codec == "MP3"
            assert result[0].bitrate == 128
            assert result[0].hls is False
            assert result[0].lastcheckok is True
            assert result[0].last_check_time == datetime(2023, 1, 1, 0, 0, 0)
            assert result[0].last_check_ok_time == datetime(2023, 1, 1, 0, 0, 0)
            assert result[0].last_local_check_time == datetime(2023, 1, 1, 0, 0, 0)
            assert result[0].click_timestamp == datetime(2023, 1, 1, 0, 0, 0)
            assert result[0].click_count == 100
            assert result[0].click_trend == 5
            assert result[0].ssl_error == 0
            assert result[0].latitude == 37.7749
            assert result[0].longitude == -122.4194
            assert result[0].has_extended_info is False
            assert result[0].iso_3166_2 == "US-CA"

    @pytest.mark.asyncio
    async def test_search_without_filter(self, radio_browser):
        """Test search method without filter."""
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value="[]")
        ) as mock_request:
            await radio_browser.search()

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
                },
            )

    @pytest.mark.asyncio
    async def test_station(self, radio_browser):
        """Test station method."""
        station_json = """
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
                "language": "english",
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
                "has_extended_info": false,
                "iso_3166_2": "US-CA"
            }
        ]
        """
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=station_json)
        ) as mock_request:
            result = await radio_browser.station(uuid="station-uuid-1")

            mock_request.assert_called_once_with(
                "stations/byuuid/station-uuid-1",
                params={
                    "hidebroken": False,
                    "limit": 1,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                },
            )
            assert isinstance(result, Station)
            assert result.uuid == "station-uuid-1"
            assert result.name == "Test Station 1"

    @pytest.mark.asyncio
    async def test_station_not_found(self, radio_browser):
        """Test station method when station is not found."""
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value="[]")
        ) as mock_request:
            result = await radio_browser.station(uuid="non-existent-uuid")

            mock_request.assert_called_once_with(
                "stations/byuuid/non-existent-uuid",
                params={
                    "hidebroken": False,
                    "limit": 1,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                },
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_stations(self, radio_browser):
        """Test stations method."""
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
                "language": "english",
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
                "has_extended_info": false,
                "iso_3166_2": "US-CA"
            }
        ]
        """
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=stations_json)
        ) as mock_request:
            result = await radio_browser.stations(
                filter_by=FilterBy.COUNTRY,
                filter_term="US",
                hide_broken=True,
                limit=10,
                offset=5,
                order=Order.NAME,
                reverse=True,
            )

            mock_request.assert_called_once_with(
                "stations/bycountry/US",
                params={
                    "hidebroken": True,
                    "limit": 10,
                    "offset": 5,
                    "order": "name",
                    "reverse": True,
                },
            )
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], Station)
            assert result[0].uuid == "station-uuid-1"
            assert result[0].name == "Test Station 1"

    @pytest.mark.asyncio
    async def test_stations_without_filter(self, radio_browser):
        """Test stations method without filter."""
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value="[]")
        ) as mock_request:
            await radio_browser.stations()

            mock_request.assert_called_once_with(
                "stations",
                params={
                    "hidebroken": False,
                    "limit": 100000,
                    "offset": 0,
                    "order": "name",
                    "reverse": False,
                },
            )

    @pytest.mark.asyncio
    async def test_tags(self, radio_browser):
        """Test tags method."""
        tags_json = """
        [
            {
                "name": "rock",
                "stationcount": "500"
            },
            {
                "name": "pop",
                "stationcount": "400"
            },
            {
                "name": "jazz",
                "stationcount": "300"
            }
        ]
        """
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value=tags_json)
        ) as mock_request:
            result = await radio_browser.tags()

            mock_request.assert_called_once_with(
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
            assert len(result) == 3
            assert all(isinstance(tag, Tag) for tag in result)
            
            # Find tags by name
            rock_tag = next((t for t in result if t.name == "rock"), None)
            pop_tag = next((t for t in result if t.name == "pop"), None)
            jazz_tag = next((t for t in result if t.name == "jazz"), None)
            
            assert rock_tag is not None
            assert rock_tag.station_count == "500"
            
            assert pop_tag is not None
            assert pop_tag.station_count == "400"
            
            assert jazz_tag is not None
            assert jazz_tag.station_count == "300"

    @pytest.mark.asyncio
    async def test_tags_with_parameters(self, radio_browser):
        """Test tags method with parameters."""
        with patch.object(
            radio_browser, "_request", AsyncMock(return_value="[]")
        ) as mock_request:
            await radio_browser.tags(
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
                },
            )

    @pytest.mark.asyncio
    async def test_close(self, radio_browser):
        """Test close method."""
        radio_browser.session = AsyncMock()
        radio_browser._close_session = True

        await radio_browser.close()

        radio_browser.session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_session(self, radio_browser):
        """Test close method with no session."""
        radio_browser.session = None
        radio_browser._close_session = True

        await radio_browser.close()  # Should not raise an exception

    @pytest.mark.asyncio
    async def test_close_not_owned_session(self, radio_browser):
        """Test close method with a session not owned by the instance."""
        radio_browser.session = AsyncMock()
        radio_browser._close_session = False

        await radio_browser.close()

        radio_browser.session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_aenter(self, radio_browser):
        """Test __aenter__ method."""
        result = await radio_browser.__aenter__()
        assert result is radio_browser

    @pytest.mark.asyncio
    async def test_aexit(self, radio_browser):
        """Test __aexit__ method."""
        radio_browser.close = AsyncMock()
        await radio_browser.__aexit__(None, None, None)
        radio_browser.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, radio_browser):
        """Test using RadioBrowser as a context manager."""
        radio_browser.close = AsyncMock()
        
        async with radio_browser as rb:
            assert rb is radio_browser
        
        radio_browser.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_backoff_retry(self, radio_browser, mock_session):
        """Test that _request retries on connection errors."""
        radio_browser._host = "api.radio-browser.info"
        radio_browser.session = mock_session
        
        # First 2 calls raise ConnectionError, third succeeds
        side_effects = [
            aiohttp.ClientError(),
            aiohttp.ClientError(),
            None
        ]
        
        def side_effect(*args, **kwargs):
            if side_effects:
                effect = side_effects.pop(0)
                if effect:
                    raise effect
            response = AsyncMock()
            response.status = 200
            response.headers = {"Content-Type": "application/json"}
            response.text = AsyncMock(return_value='{"test": "success"}')
            return response
        
        mock_session.request.side_effect = side_effect
        
        with patch("radios.radio_browser.DNSResolver") as mock_resolver_class:
            mock_resolver = AsyncMock()
            mock_result = [MagicMock(host="api.radio-browser.info")]
            mock_resolver.query.return_value = mock_result
            mock_resolver_class.return_value = mock_resolver
            
            result = await radio_browser._request("test")
            
            assert result == '{"test": "success"}'
            assert mock_session.request.call_count == 3