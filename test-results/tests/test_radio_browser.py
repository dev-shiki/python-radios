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
def radio_browser():
    return RadioBrowser(user_agent="Test User Agent")


@pytest.fixture
def mock_dns_resolver():
    resolver = MagicMock(spec=DNSResolver)
    resolver_result = MagicMock()
    resolver_result.host = "api.radio-browser.info"
    resolver.query.return_value = asyncio.Future()
    resolver.query.return_value.set_result([resolver_result])
    return resolver


@pytest.fixture
def mock_session():
    session = AsyncMock(spec=aiohttp.ClientSession)
    response = AsyncMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    session.request.return_value = response
    return session


@pytest.mark.asyncio
async def test_request_dns_resolution(radio_browser, mock_dns_resolver):
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch.object(radio_browser, "session", AsyncMock()):
            radio_browser.session.request.return_value.__aenter__.return_value.text = AsyncMock(
                return_value="{}"
            )
            radio_browser.session.request.return_value.__aenter__.return_value.headers = {
                "Content-Type": "application/json"
            }
            
            await radio_browser._request("test")
            
            mock_dns_resolver.query.assert_called_once_with(
                "_api._tcp.radio-browser.info", "SRV"
            )
            assert radio_browser._host == "api.radio-browser.info"


@pytest.mark.asyncio
async def test_request_creates_session_if_none(radio_browser, mock_dns_resolver):
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch("radios.radio_browser.aiohttp.ClientSession") as mock_client_session:
            mock_session = AsyncMock()
            mock_client_session.return_value = mock_session
            mock_session.request.return_value.__aenter__.return_value.text = AsyncMock(
                return_value="{}"
            )
            mock_session.request.return_value.__aenter__.return_value.headers = {
                "Content-Type": "application/json"
            }
            
            await radio_browser._request("test")
            
            mock_client_session.assert_called_once()
            assert radio_browser.session == mock_session
            assert radio_browser._close_session is True


@pytest.mark.asyncio
async def test_request_uses_existing_session(radio_browser, mock_dns_resolver):
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        mock_session = AsyncMock()
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        mock_session.request.return_value.__aenter__.return_value.text = AsyncMock(
            return_value="{}"
        )
        mock_session.request.return_value.__aenter__.return_value.headers = {
            "Content-Type": "application/json"
        }
        
        await radio_browser._request("test")
        
        mock_session.request.assert_called_once()


@pytest.mark.asyncio
async def test_request_timeout_error(radio_browser, mock_dns_resolver):
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        mock_session = AsyncMock()
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        mock_session.request.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(RadioBrowserConnectionTimeoutError):
            await radio_browser._request("test")
        
        assert radio_browser._host is None


@pytest.mark.asyncio
async def test_request_connection_error(radio_browser, mock_dns_resolver):
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        mock_session = AsyncMock()
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        mock_session.request.side_effect = aiohttp.ClientError()
        
        with pytest.raises(RadioBrowserConnectionError):
            await radio_browser._request("test")
        
        assert radio_browser._host is None


@pytest.mark.asyncio
async def test_request_socket_error(radio_browser, mock_dns_resolver):
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        mock_session = AsyncMock()
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        mock_session.request.side_effect = socket.gaierror()
        
        with pytest.raises(RadioBrowserConnectionError):
            await radio_browser._request("test")
        
        assert radio_browser._host is None


@pytest.mark.asyncio
async def test_request_invalid_content_type(radio_browser, mock_dns_resolver):
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        mock_session = AsyncMock()
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.text.return_value = "Not JSON"
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(RadioBrowserError):
            await radio_browser._request("test")


@pytest.mark.asyncio
async def test_request_boolean_params(radio_browser, mock_dns_resolver):
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        mock_session = AsyncMock()
        radio_browser.session = mock_session
        radio_browser._host = "api.radio-browser.info"
        mock_session.request.return_value.__aenter__.return_value.text = AsyncMock(
            return_value="{}"
        )
        mock_session.request.return_value.__aenter__.return_value.headers = {
            "Content-Type": "application/json"
        }
        
        await radio_browser._request("test", params={"bool_param": True})
        
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args[1]
        assert call_args["params"]["bool_param"] == "true"


@pytest.mark.asyncio
async def test_stats(radio_browser):
    stats_json = """
    {
        "supported_version": 1,
        "software_version": "1.0.0",
        "status": "OK",
        "stations": 1000,
        "stations_broken": 10,
        "tags": 500,
        "clicks_last_hour": 100,
        "clicks_last_day": 2400,
        "languages": 50,
        "countries": 100
    }
    """
    
    with patch.object(radio_browser, "_request", AsyncMock(return_value=stats_json)):
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
        assert result.clicks_last_day == 2400
        assert result.languages == 50
        assert result.countries == 100


@pytest.mark.asyncio
async def test_station_click(radio_browser):
    with patch.object(radio_browser, "_request", AsyncMock(return_value="{}")):
        await radio_browser.station_click(uuid="test-uuid")
        
        radio_browser._request.assert_called_once_with("url/test-uuid")


@pytest.mark.asyncio
async def test_countries(radio_browser):
    countries_json = """
    [
        {
            "name": "US",
            "stationcount": "500"
        },
        {
            "name": "XK",
            "stationcount": "50"
        },
        {
            "name": "DE",
            "stationcount": "300"
        }
    ]
    """
    
    with patch.object(radio_browser, "_request", AsyncMock(return_value=countries_json)):
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
        
        assert len(result) == 3
        assert isinstance(result[0], Country)
        
        # Find countries by code
        us_country = next((c for c in result if c.code == "US"), None)
        kosovo_country = next((c for c in result if c.code == "XK"), None)
        de_country = next((c for c in result if c.code == "DE"), None)
        
        assert us_country is not None
        assert us_country.name == "United States"
        assert us_country.station_count == "500"
        
        assert kosovo_country is not None
        assert kosovo_country.name == "Kosovo"
        assert kosovo_country.station_count == "50"
        
        assert de_country is not None
        assert de_country.name == "Germany"
        assert de_country.station_count == "300"


@pytest.mark.asyncio
async def test_countries_with_parameters(radio_browser):
    with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")):
        await radio_browser.countries(
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


@pytest.mark.asyncio
async def test_languages(radio_browser):
    languages_json = """
    [
        {
            "name": "english",
            "iso_639": "en",
            "stationcount": "1000"
        },
        {
            "name": "german",
            "iso_639": "de",
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
        
        assert len(result) == 3
        assert isinstance(result[0], Language)
        
        # Find languages by code
        en_language = next((l for l in result if l.code == "en"), None)
        de_language = next((l for l in result if l.code == "de"), None)
        es_language = next((l for l in result if l.code == "es"), None)
        
        assert en_language is not None
        assert en_language.name == "English"
        assert en_language.station_count == "1000"
        
        assert de_language is not None
        assert de_language.name == "German"
        assert de_language.station_count == "500"
        
        assert es_language is not None
        assert es_language.name == "Spanish"
        assert es_language.station_count == "300"


@pytest.mark.asyncio
async def test_languages_with_parameters(radio_browser):
    with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")):
        await radio_browser.languages(
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


@pytest.mark.asyncio
async def test_search(radio_browser):
    stations_json = """
    [
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
            "has_extended_info": true,
            "iso_3166_2": "US-CA"
        }
    ]
    """
    
    with patch.object(radio_browser, "_request", AsyncMock(return_value=stations_json)):
        result = await radio_browser.search(
            filter_by=FilterBy.NAME,
            filter_term="test",
            name="Test Station",
            country="US",
            bitrate_min=128,
            bitrate_max=320,
        )
        
        radio_browser._request.assert_called_once_with(
            "stations/search/byname/test",
            params={
                "hidebroken": False,
                "limit": 100000,
                "offset": 0,
                "order": "name",
                "reverse": False,
                "name": "Test Station",
                "name_exact": False,
                "country": "US",
                "country_exact": False,
                "state_exact": False,
                "language_exact": False,
                "tag_exact": False,
                "bitrate_min": 128,
                "bitrate_max": 320,
            },
        )
        
        assert len(result) == 1
        assert isinstance(result[0], Station)
        assert result[0].uuid == "station-uuid-1"
        assert result[0].name == "Test Station 1"
        assert result[0].url == "https://example.com/stream1"
        assert result[0].bitrate == 128
        assert result[0].country_code == "US"
        assert result[0].country == "United States"


@pytest.mark.asyncio
async def test_search_without_filter(radio_browser):
    with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")):
        await radio_browser.search()
        
        radio_browser._request.assert_called_once_with(
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
async def test_station(radio_browser):
    station_json = """
    [
        {
            "changeuuid": "change-uuid-1",
            "stationuuid": "test-uuid",
            "name": "Test Station",
            "url": "https://example.com/stream",
            "url_resolved": "https://example.com/stream",
            "homepage": "https://example.com",
            "favicon": "https://example.com/favicon.ico",
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
            "has_extended_info": true,
            "iso_3166_2": "US-CA"
        }
    ]
    """
    
    with patch.object(radio_browser, "_request", AsyncMock(return_value=station_json)):
        result = await radio_browser.station(uuid="test-uuid")
        
        assert radio_browser._request.call_count == 0  # It should use stations() method
        assert isinstance(result, Station)
        assert result.uuid == "test-uuid"
        assert result.name == "Test Station"


@pytest.mark.asyncio
async def test_station_not_found(radio_browser):
    with patch.object(radio_browser, "stations", AsyncMock(return_value=[])):
        result = await radio_browser.station(uuid="non-existent-uuid")
        
        radio_browser.stations.assert_called_once_with(
            filter_by=FilterBy.UUID,
            filter_term="non-existent-uuid",
            limit=1,
        )
        assert result is None


@pytest.mark.asyncio
async def test_stations(radio_browser):
    stations_json = """
    [
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
            "has_extended_info": true,
            "iso_3166_2": "US-CA"
        }
    ]
    """
    
    with patch.object(radio_browser, "_request", AsyncMock(return_value=stations_json)):
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
        
        assert len(result) == 1
        assert isinstance(result[0], Station)
        assert result[0].uuid == "station-uuid-1"
        assert result[0].name == "Test Station 1"


@pytest.mark.asyncio
async def test_stations_with_filter(radio_browser):
    with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")):
        await radio_browser.stations(
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


@pytest.mark.asyncio
async def test_tags(radio_browser):
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
        
        assert len(result) == 3
        assert isinstance(result[0], Tag)
        
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
async def test_tags_with_parameters(radio_browser):
    with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")):
        await radio_browser.tags(
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


@pytest.mark.asyncio
async def test_close(radio_browser):
    mock_session = AsyncMock()
    radio_browser.session = mock_session
    radio_browser._close_session = True
    
    await radio_browser.close()
    
    mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_no_session(radio_browser):
    radio_browser.session = None
    
    await radio_browser.close()  # Should not raise an exception


@pytest.mark.asyncio
async def test_close_not_owned_session(radio_browser):
    mock_session = AsyncMock()
    radio_browser.session = mock_session
    radio_browser._close_session = False
    
    await radio_browser.close()
    
    mock_session.close.assert_not_called()


@pytest.mark.asyncio
async def test_aenter(radio_browser):
    result = await radio_browser.__aenter__()
    assert result is radio_browser


@pytest.mark.asyncio
async def test_aexit(radio_browser):
    with patch.object(radio_browser, "close", AsyncMock()) as mock_close:
        await radio_browser.__aexit__(None, None, None)
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_context_manager():
    with patch("radios.radio_browser.RadioBrowser.close", AsyncMock()):
        async with RadioBrowser(user_agent="Test User Agent") as rb:
            assert isinstance(rb, RadioBrowser)
            assert rb.user_agent == "Test User Agent"
        
        rb.close.assert_called_once()