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
    response = AsyncMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    response.text = AsyncMock()
    session.request.return_value = response
    return session


@pytest.fixture
def mock_dns_resolver():
    """Return a mock DNSResolver."""
    resolver = AsyncMock()
    result = [MagicMock()]
    result[0].host = "api.radio-browser.info"
    resolver.query.return_value = result
    return resolver


@pytest.mark.asyncio
async def test_request_dns_resolution(radio_browser, mock_dns_resolver):
    """Test DNS resolution in _request method."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.text.return_value = "{}"
            mock_session.request.return_value = mock_response
            mock_session_class.return_value = mock_session

            await radio_browser._request("test")

            mock_dns_resolver.query.assert_called_once_with(
                "_api._tcp.radio-browser.info", "SRV"
            )
            assert radio_browser._host == "api.radio-browser.info"


@pytest.mark.asyncio
async def test_request_session_creation(radio_browser):
    """Test session creation in _request method."""
    with patch("radios.radio_browser.DNSResolver") as mock_resolver_class:
        mock_resolver = AsyncMock()
        result = [MagicMock()]
        result[0].host = "api.radio-browser.info"
        mock_resolver.query.return_value = result
        mock_resolver_class.return_value = mock_resolver

        with patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.text.return_value = "{}"
            mock_session.request.return_value = mock_response
            mock_session_class.return_value = mock_session

            assert radio_browser.session is None
            await radio_browser._request("test")
            assert radio_browser.session is not None
            assert radio_browser._close_session is True


@pytest.mark.asyncio
async def test_request_with_params(radio_browser, mock_dns_resolver):
    """Test _request method with parameters."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.text.return_value = "{}"
            mock_session.request.return_value = mock_response
            mock_session_class.return_value = mock_session

            await radio_browser._request("test", params={"test": True, "value": "test"})

            mock_session.request.assert_called_once()
            call_args = mock_session.request.call_args[1]
            assert call_args["params"] == {"test": "true", "value": "test"}


@pytest.mark.asyncio
async def test_request_timeout_error(radio_browser, mock_dns_resolver):
    """Test _request method with timeout error."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.request.side_effect = asyncio.TimeoutError()
            mock_session_class.return_value = mock_session

            with pytest.raises(RadioBrowserConnectionTimeoutError):
                await radio_browser._request("test")
            
            assert radio_browser._host is None


@pytest.mark.asyncio
async def test_request_client_error(radio_browser, mock_dns_resolver):
    """Test _request method with client error."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.request.side_effect = aiohttp.ClientError()
            mock_session_class.return_value = mock_session

            with pytest.raises(RadioBrowserConnectionError):
                await radio_browser._request("test")
            
            assert radio_browser._host is None


@pytest.mark.asyncio
async def test_request_socket_error(radio_browser, mock_dns_resolver):
    """Test _request method with socket error."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.request.side_effect = socket.gaierror()
            mock_session_class.return_value = mock_session

            with pytest.raises(RadioBrowserConnectionError):
                await radio_browser._request("test")
            
            assert radio_browser._host is None


@pytest.mark.asyncio
async def test_request_invalid_content_type(radio_browser, mock_dns_resolver):
    """Test _request method with invalid content type."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.text.return_value = "<html>Error</html>"
            mock_response.status = 200
            mock_session.request.return_value = mock_response
            mock_session_class.return_value = mock_session

            with pytest.raises(RadioBrowserError):
                await radio_browser._request("test")


@pytest.mark.asyncio
async def test_stats(radio_browser):
    """Test stats method."""
    stats_data = {
        "supported_version": 1,
        "software_version": "2.0.0",
        "status": "OK",
        "stations": 30000,
        "stations_broken": 1000,
        "tags": 5000,
        "clicks_last_hour": 10000,
        "clicks_last_day": 50000,
        "languages": 100,
        "countries": 200,
    }
    
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps(stats_data)
        
        result = await radio_browser.stats()
        
        mock_request.assert_called_once_with("stats")
        assert isinstance(result, Stats)
        assert result.supported_version == 1
        assert result.software_version == AwesomeVersion("2.0.0")
        assert result.status == "OK"
        assert result.stations == 30000
        assert result.stations_broken == 1000
        assert result.tags == 5000
        assert result.clicks_last_hour == 10000
        assert result.clicks_last_day == 50000
        assert result.languages == 100
        assert result.countries == 200


@pytest.mark.asyncio
async def test_station_click(radio_browser):
    """Test station_click method."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = "{}"
        
        await radio_browser.station_click(uuid="test-uuid")
        
        mock_request.assert_called_once_with("url/test-uuid")


@pytest.mark.asyncio
async def test_countries(radio_browser):
    """Test countries method."""
    countries_data = [
        {"name": "US", "stationcount": "1000"},
        {"name": "GB", "stationcount": "500"},
        {"name": "XK", "stationcount": "50"},  # Kosovo special case
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
            },
        )
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(country, Country) for country in result)
        
        # Check Kosovo special case
        kosovo = next(country for country in result if country.code == "XK")
        assert kosovo.name == "Kosovo"
        
        # Check normal country resolution
        us = next(country for country in result if country.code == "US")
        assert us.name == "United States"


@pytest.mark.asyncio
async def test_countries_with_parameters(radio_browser):
    """Test countries method with custom parameters."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([])
        
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
async def test_languages(radio_browser):
    """Test languages method."""
    languages_data = [
        {"name": "english", "stationcount": "1000", "iso_639": "en"},
        {"name": "spanish", "stationcount": "500", "iso_639": "es"},
        {"name": "unknown", "stationcount": "50", "iso_639": None},
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
            },
        )
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(language, Language) for language in result)
        
        # Check title case conversion
        english = next(language for language in result if language.code == "en")
        assert english.name == "English"
        
        # Check language with no code
        unknown = next(language for language in result if language.code is None)
        assert unknown.name == "Unknown"
        assert unknown.favicon is None


@pytest.mark.asyncio
async def test_languages_with_parameters(radio_browser):
    """Test languages method with custom parameters."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([])
        
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
async def test_search(radio_browser):
    """Test search method."""
    stations_data = [
        {
            "changeuuid": "change-uuid-1",
            "stationuuid": "station-uuid-1",
            "name": "Test Station 1",
            "url": "https://example.com/stream1",
            "url_resolved": "https://example.com/stream1",
            "homepage": "https://example.com",
            "favicon": "https://example.com/favicon.ico",
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
            "clicktrend": 5,
            "ssl_error": 0,
            "geo_lat": 37.7749,
            "geo_long": -122.4194,
            "has_extended_info": True,
            "iso_3166_2": "US-CA"
        }
    ]
    
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps(stations_data)
        
        result = await radio_browser.search(name="test")
        
        mock_request.assert_called_once()
        assert mock_request.call_args[0][0] == "stations/search"
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Station)
        assert result[0].name == "Test Station 1"
        assert result[0].uuid == "station-uuid-1"


@pytest.mark.asyncio
async def test_search_with_filter(radio_browser):
    """Test search method with filter."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([])
        
        await radio_browser.search(
            filter_by=FilterBy.COUNTRY,
            filter_term="US",
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.NAME,
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
        
        mock_request.assert_called_once()
        assert mock_request.call_args[0][0] == "stations/search/bycountry/US"
        params = mock_request.call_args[1]["params"]
        assert params["hidebroken"] is True
        assert params["limit"] == 10
        assert params["offset"] == 5
        assert params["order"] == "name"
        assert params["reverse"] is True
        assert params["name"] == "test"
        assert params["name_exact"] is True
        assert params["country"] == "US"
        assert params["country_exact"] is True
        assert params["state_exact"] is True
        assert params["language_exact"] is True
        assert params["tag_exact"] is True
        assert params["bitrate_min"] == 128
        assert params["bitrate_max"] == 320


@pytest.mark.asyncio
async def test_station(radio_browser):
    """Test station method."""
    station_data = [
        {
            "changeuuid": "change-uuid-1",
            "stationuuid": "station-uuid-1",
            "name": "Test Station 1",
            "url": "https://example.com/stream1",
            "url_resolved": "https://example.com/stream1",
            "homepage": "https://example.com",
            "favicon": "https://example.com/favicon.ico",
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
            "clicktrend": 5,
            "ssl_error": 0,
            "geo_lat": 37.7749,
            "geo_long": -122.4194,
            "has_extended_info": True,
            "iso_3166_2": "US-CA"
        }
    ]
    
    with patch.object(radio_browser, "stations") as mock_stations:
        station = Station.from_dict(station_data[0])
        mock_stations.return_value = [station]
        
        result = await radio_browser.station(uuid="station-uuid-1")
        
        mock_stations.assert_called_once_with(
            filter_by=FilterBy.UUID,
            filter_term="station-uuid-1",
            limit=1,
        )
        
        assert result is station


@pytest.mark.asyncio
async def test_station_not_found(radio_browser):
    """Test station method when station is not found."""
    with patch.object(radio_browser, "stations") as mock_stations:
        mock_stations.return_value = []
        
        result = await radio_browser.station(uuid="non-existent")
        
        assert result is None


@pytest.mark.asyncio
async def test_stations(radio_browser):
    """Test stations method."""
    stations_data = [
        {
            "changeuuid": "change-uuid-1",
            "stationuuid": "station-uuid-1",
            "name": "Test Station 1",
            "url": "https://example.com/stream1",
            "url_resolved": "https://example.com/stream1",
            "homepage": "https://example.com",
            "favicon": "https://example.com/favicon.ico",
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
            "clicktrend": 5,
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
            },
        )
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Station)
        assert result[0].name == "Test Station 1"
        assert result[0].uuid == "station-uuid-1"


@pytest.mark.asyncio
async def test_stations_with_filter(radio_browser):
    """Test stations method with filter."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([])
        
        await radio_browser.stations(
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


@pytest.mark.asyncio
async def test_tags(radio_browser):
    """Test tags method."""
    tags_data = [
        {"name": "rock", "stationcount": "1000"},
        {"name": "pop", "stationcount": "500"},
        {"name": "jazz", "stationcount": "200"},
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
            },
        )
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(tag, Tag) for tag in result)
        
        rock = next(tag for tag in result if tag.name == "rock")
        assert rock.station_count == "1000"


@pytest.mark.asyncio
async def test_tags_with_parameters(radio_browser):
    """Test tags method with custom parameters."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([])
        
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
async def test_close(radio_browser):
    """Test close method."""
    radio_browser.session = AsyncMock()
    radio_browser._close_session = True
    
    await radio_browser.close()
    
    radio_browser.session.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_no_session(radio_browser):
    """Test close method with no session."""
    radio_browser.session = None
    
    await radio_browser.close()  # Should not raise an exception


@pytest.mark.asyncio
async def test_close_not_owned_session(radio_browser):
    """Test close method with a session that should not be closed."""
    radio_browser.session = AsyncMock()
    radio_browser._close_session = False
    
    await radio_browser.close()
    
    radio_browser.session.close.assert_not_called()


@pytest.mark.asyncio
async def test_aenter(radio_browser):
    """Test __aenter__ method."""
    result = await radio_browser.__aenter__()
    assert result is radio_browser


@pytest.mark.asyncio
async def test_aexit(radio_browser):
    """Test __aexit__ method."""
    radio_browser.close = AsyncMock()
    
    await radio_browser.__aexit__(None, None, None)
    
    radio_browser.close.assert_called_once()


@pytest.mark.asyncio
async def test_context_manager():
    """Test RadioBrowser as a context manager."""
    with patch("radios.radio_browser.RadioBrowser.close") as mock_close:
        mock_close.return_value = None
        
        async with RadioBrowser(user_agent="Test/1.0") as rb:
            assert isinstance(rb, RadioBrowser)
        
        mock_close.assert_called_once()