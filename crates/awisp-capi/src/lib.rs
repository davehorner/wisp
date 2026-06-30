use bevy::app::App;
use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::window::{
    ExitCondition, PrimaryWindow, WindowCloseRequested, WindowLevel, WindowPlugin, WindowPosition,
    WindowRef, WindowResolution,
};
use bevy_wisp::asset::Wisp;
use bevy_wisp::prelude::*;
use bevy_wisp::schema::TextureRole;
use rosc::{OscMessage, OscPacket, OscType, encoder};
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, c_char};
use std::fs;
use std::net::{SocketAddr, UdpSocket};
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

const SHADERS: &[(&str, &str)] = &[
    ("wisp/test_audio.wgsl", "test audio"),
    ("wisp/test_audio_fft.wgsl", "test audio fft"),
    ("wisp/test_color.wgsl", "test color"),
    ("wisp/test_compute.wgsl", "test compute"),
    ("wisp/test_float.wgsl", "test float"),
    ("wisp/test_image.wgsl", "test image"),
    ("wisp/test_inputs.wgsl", "test inputs"),
    ("wisp/test_multi_pass_rendering.wgsl", "test multi pass"),
    ("wisp/test_persistent_buffer.wgsl", "test persistent buffer"),
];

#[derive(Clone, Debug)]
enum Command {
    OpenWindow {
        shader_name: String,
        window_title: String,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    },
    LoadShader(String),
    SetVisible(bool),
    SetGeometry {
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    },
    PushAudio {
        interleaved: Vec<f32>,
        channels: usize,
    },
    LoadImage(PathBuf),
    SetTitleBarVisible(bool),
    SetParam {
        id: String,
        value: ParamValue,
    },
}

#[derive(Clone, Debug)]
struct ControlCommand {
    instance_id: u64,
    command: Command,
}

#[derive(Clone, Debug)]
enum ParamValue {
    Float(Vec<f32>),
    I32(i32),
    U32(u32),
    Bool(bool),
}

#[derive(Clone, Debug)]
enum RemoteMessage {
    Command(Command),
    Discover,
    Subscribe { enabled: bool },
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum AwispParamType {
    F32 = 0,
    I32 = 1,
    U32 = 2,
    Bool = 3,
    Vec2 = 4,
    Vec3 = 5,
    Vec4 = 6,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AwispParamDesc {
    id: *const c_char,
    label: *const c_char,
    param_type: AwispParamType,
    component_count: u32,
    min_value: f32,
    max_value: f32,
    step: f32,
    default_values: [f32; 4],
    is_color: bool,
    option_count: usize,
}

const STATUS_STARTING: u8 = 0;
const STATUS_RUNNING: u8 = 1;
const STATUS_EXITED: u8 = 2;
const STATUS_PANICKED: u8 = 3;

#[derive(Resource)]
struct ControlReceiver(Mutex<Receiver<ControlCommand>>);

#[derive(Resource, Clone)]
struct StartupShader {
    instance_id: u64,
    shader_name: String,
}

#[derive(Resource, Default)]
struct PendingParams(Vec<(u64, String, ParamValue)>);

#[derive(Resource, Default)]
struct AwispAudioFeeds(HashMap<u64, WispAudio>);

#[derive(Resource, Default)]
struct DefaultImageInputs(HashMap<String, Handle<Image>>);

#[derive(Component, Clone, Copy)]
struct AwispSurface {
    instance_id: u64,
}

struct CachedParam {
    id: CString,
    label: CString,
    param_type: AwispParamType,
    component_count: u32,
    min_value: f32,
    max_value: f32,
    step: f32,
    default_values: [f32; 4],
    is_color: bool,
    options: Vec<(i32, CString)>,
}

struct ParamCache {
    key: String,
    params: Vec<CachedParam>,
}

pub struct AwispInstance {
    instance_id: u64,
    sender: Sender<ControlCommand>,
    status: Arc<AtomicU8>,
    error: Arc<Mutex<CString>>,
    asset_root: Arc<Mutex<String>>,
    current_shader: Arc<Mutex<String>>,
    feedback_target: Arc<Mutex<Option<SocketAddr>>>,
    feedback_enabled: Arc<Mutex<bool>>,
}

struct SharedAwispRuntime {
    sender: Sender<ControlCommand>,
    status: Arc<AtomicU8>,
    error: Arc<Mutex<CString>>,
    asset_root: Arc<Mutex<String>>,
    current_shader: Arc<Mutex<String>>,
    feedback_target: Arc<Mutex<Option<SocketAddr>>>,
    feedback_enabled: Arc<Mutex<bool>>,
}

fn last_error_cell() -> &'static Mutex<CString> {
    static LAST_ERROR: OnceLock<Mutex<CString>> = OnceLock::new();
    LAST_ERROR.get_or_init(|| Mutex::new(CString::new("").unwrap()))
}

fn set_last_error(message: impl Into<String>) {
    let sanitized = sanitize_error(message);
    if let Ok(mut lock) = last_error_cell().lock() {
        *lock = CString::new(sanitized).unwrap_or_else(|_| CString::new("awisp error").unwrap());
    }
}

fn sanitize_error(message: impl Into<String>) -> String {
    message.into().replace('\0', " ")
}

fn set_instance_error(error: &Mutex<CString>, message: impl Into<String>) {
    let sanitized = sanitize_error(message);
    if let Ok(mut lock) = error.lock() {
        *lock = CString::new(sanitized).unwrap_or_else(|_| CString::new("awisp error").unwrap());
    }
}

fn param_cache_cell() -> &'static Mutex<Option<ParamCache>> {
    static PARAM_CACHE: OnceLock<Mutex<Option<ParamCache>>> = OnceLock::new();
    PARAM_CACHE.get_or_init(|| Mutex::new(None))
}

fn embedded_runtime_cell() -> &'static Mutex<Option<SharedAwispRuntime>> {
    static EMBEDDED_RUNTIME: OnceLock<Mutex<Option<SharedAwispRuntime>>> = OnceLock::new();
    EMBEDDED_RUNTIME.get_or_init(|| Mutex::new(None))
}

fn remote_ports_cell() -> &'static Mutex<HashSet<u16>> {
    static REMOTE_PORTS: OnceLock<Mutex<HashSet<u16>>> = OnceLock::new();
    REMOTE_PORTS.get_or_init(|| Mutex::new(HashSet::new()))
}

fn next_instance_id() -> u64 {
    static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
}

unsafe fn opt_cstr(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned(),
    )
}

fn shader_path(asset_root: &str, shader_name: &str) -> PathBuf {
    let shader_path = Path::new(shader_name);
    if shader_path.is_absolute() {
        shader_path.to_path_buf()
    } else {
        Path::new(asset_root).join(shader_name)
    }
}

fn component_count(ty: &bevy_wisp::schema::ParamType) -> u32 {
    match ty {
        bevy_wisp::schema::ParamType::Vec2 => 2,
        bevy_wisp::schema::ParamType::Vec3 => 3,
        bevy_wisp::schema::ParamType::Vec4 => 4,
        _ => 1,
    }
}

fn capi_param_type(ty: &bevy_wisp::schema::ParamType) -> AwispParamType {
    match ty {
        bevy_wisp::schema::ParamType::F32 => AwispParamType::F32,
        bevy_wisp::schema::ParamType::I32 => AwispParamType::I32,
        bevy_wisp::schema::ParamType::U32 => AwispParamType::U32,
        bevy_wisp::schema::ParamType::Bool => AwispParamType::Bool,
        bevy_wisp::schema::ParamType::Vec2 => AwispParamType::Vec2,
        bevy_wisp::schema::ParamType::Vec3 => AwispParamType::Vec3,
        bevy_wisp::schema::ParamType::Vec4 => AwispParamType::Vec4,
    }
}

fn default_range(ty: &bevy_wisp::schema::ParamType, defaults: &[f32; 4]) -> (f32, f32) {
    match ty {
        bevy_wisp::schema::ParamType::Bool => (0.0, 1.0),
        bevy_wisp::schema::ParamType::I32 | bevy_wisp::schema::ParamType::U32 => {
            (defaults[0].min(0.0), defaults[0].max(100.0))
        }
        _ => {
            let mut min_value = -1.0f32;
            let mut max_value = 1.0f32;
            for value in defaults {
                min_value = min_value.min(*value);
                max_value = max_value.max(*value);
            }
            (min_value, max_value)
        }
    }
}

fn load_param_cache(asset_root: &str, shader_name: &str) -> Result<ParamCache, String> {
    let path = shader_path(asset_root, shader_name);
    let source = fs::read_to_string(&path)
        .map_err(|error| format!("failed to read awisp shader {}: {error}", path.display()))?;
    let reflected = bevy_wisp::reflect::parse_and_validate(&source)
        .map_err(|error| format!("failed to parse awisp shader {}: {error}", path.display()))?;
    let schema = bevy_wisp::schema::schema_from_module(&reflected)
        .map_err(|error| format!("failed to reflect awisp shader {}: {error}", path.display()))?;
    let Some(params) = schema.params else {
        return Ok(ParamCache {
            key: format!("{asset_root}\0{shader_name}"),
            params: Vec::new(),
        });
    };

    let mut cached = Vec::with_capacity(params.fields.len());
    for field in params.fields {
        let component_count = component_count(&field.ty);
        let mut default_values = [0.0f32; 4];
        if let Some(default) = &field.ui.default {
            for (index, value) in default.iter().take(4).enumerate() {
                default_values[index] = *value as f32;
            }
        } else if matches!(field.ty, bevy_wisp::schema::ParamType::Bool) {
            default_values[0] = 0.0;
        }
        let (fallback_min, fallback_max) = default_range(&field.ty, &default_values);
        let min_value = field.ui.min.map_or(fallback_min, |value| value as f32);
        let max_value = field.ui.max.map_or(fallback_max, |value| value as f32);
        let label = field
            .ui
            .label
            .clone()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| field.name.clone());
        let options = field
            .ui
            .values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let label = field
                    .ui
                    .labels
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| value.to_string());
                (
                    (*value).clamp(i32::MIN as i64, i32::MAX as i64) as i32,
                    CString::new(label).unwrap_or_else(|_| CString::new("option").unwrap()),
                )
            })
            .collect();
        cached.push(CachedParam {
            id: CString::new(field.name).unwrap_or_else(|_| CString::new("param").unwrap()),
            label: CString::new(label).unwrap_or_else(|_| CString::new("param").unwrap()),
            param_type: capi_param_type(&field.ty),
            component_count,
            min_value,
            max_value: max_value.max(min_value + 0.0001),
            step: field.ui.step.map_or(0.0, |value| value as f32),
            default_values,
            is_color: field.ui.color,
            options,
        });
    }

    Ok(ParamCache {
        key: format!("{asset_root}\0{shader_name}"),
        params: cached,
    })
}

fn ensure_param_cache(asset_root: &str, shader_name: &str) -> Result<(), String> {
    let key = format!("{asset_root}\0{shader_name}");
    let cell = param_cache_cell();
    {
        let cache = cell
            .lock()
            .map_err(|_| "awisp param cache poisoned".to_string())?;
        if cache.as_ref().is_some_and(|cache| cache.key == key) {
            return Ok(());
        }
    }
    let loaded = load_param_cache(asset_root, shader_name)?;
    let mut cache = cell
        .lock()
        .map_err(|_| "awisp param cache poisoned".to_string())?;
    *cache = Some(loaded);
    Ok(())
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    startup: Res<StartupShader>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
) {
    let wisp: Handle<Wisp> = asset_server.load(startup.shader_name.clone());
    let Ok(window_entity) = primary_window.single() else {
        return;
    };
    commands.entity(window_entity).insert(AwispSurface {
        instance_id: startup.instance_id,
    });
    commands.spawn((
        Camera3d::default(),
        Camera::default(),
        RenderTarget::Window(WindowRef::Entity(window_entity)),
        WispHandle(wisp),
        AwispSurface {
            instance_id: startup.instance_id,
        },
    ));
}

fn apply_param(inputs: &mut WispInputs, id: String, value: ParamValue) {
    let value = match value {
        ParamValue::Float(values) => match values.len() {
            0 => return,
            1 => WispValue::F32(values[0]),
            2 => WispValue::Vec2(Vec2::new(values[0], values[1])),
            3 => WispValue::Vec3(Vec3::new(values[0], values[1], values[2])),
            _ => WispValue::Vec4(Vec4::new(values[0], values[1], values[2], values[3])),
        },
        ParamValue::I32(value) => WispValue::I32(value),
        ParamValue::U32(value) => WispValue::U32(value),
        ParamValue::Bool(value) => WispValue::Bool(value),
    };
    inputs.insert(id, value);
}

fn default_image_input(name: &str) -> Image {
    let width = 256u32;
    let height = 256u32;
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    let name_hash = name.bytes().fold(2166136261u32, |hash, byte| {
        (hash ^ byte as u32).wrapping_mul(16777619)
    });
    let accent = [
        80u8.saturating_add((name_hash & 0x7f) as u8),
        80u8.saturating_add(((name_hash >> 8) & 0x7f) as u8),
        80u8.saturating_add(((name_hash >> 16) & 0x7f) as u8),
    ];
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / 32) + (y / 32)) % 2 == 0;
            let stripe = ((x + y) / 16) % 2 == 0;
            let base = if checker { 230 } else { 38 };
            let mix = if stripe { 0.72 } else { 0.45 };
            data.push((base as f32 * (1.0 - mix) + accent[0] as f32 * mix) as u8);
            data.push((base as f32 * (1.0 - mix) + accent[1] as f32 * mix) as u8);
            data.push((base as f32 * (1.0 - mix) + accent[2] as f32 * mix) as u8);
            data.push(255);
        }
    }
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image
}

fn image_from_path(path: &Path) -> Result<Image, String> {
    let decoded = image::open(path)
        .map_err(|error| format!("failed to decode image {}: {error}", path.display()))?
        .to_rgba8();
    let (width, height) = decoded.dimensions();
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        decoded.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    Ok(image)
}

fn parse_bool_text(value: &str) -> bool {
    !matches!(
        value.to_ascii_lowercase().as_str(),
        "0" | "false" | "off" | "no"
    )
}

fn parse_text_command(text: &str) -> Option<Command> {
    let mut parts = text.split_whitespace();
    match parts.next()? {
        "shader" | "load_shader" => Some(Command::LoadShader(parts.collect::<Vec<_>>().join(" "))),
        "visible" | "show" => Some(Command::SetVisible(
            parts.next().map_or(true, parse_bool_text),
        )),
        "hide" => Some(Command::SetVisible(false)),
        "titlebar" | "title_bar_visible" => Some(Command::SetTitleBarVisible(
            parts.next().map_or(true, parse_bool_text),
        )),
        "hide_titlebar" | "hide_title_bar" => Some(Command::SetTitleBarVisible(false)),
        "window" | "geometry" => Some(Command::SetGeometry {
            x: parts.next()?.parse().ok()?,
            y: parts.next()?.parse().ok()?,
            width: parts.next()?.parse().ok()?,
            height: parts.next()?.parse().ok()?,
        }),
        "image" | "load_image" | "load_media" => Some(Command::LoadImage(PathBuf::from(
            parts.collect::<Vec<_>>().join(" "),
        ))),
        "set_f32" => Some(Command::SetParam {
            id: parts.next()?.to_string(),
            value: ParamValue::Float(vec![parts.next()?.parse().ok()?]),
        }),
        "set_vec2" => Some(Command::SetParam {
            id: parts.next()?.to_string(),
            value: ParamValue::Float(vec![
                parts.next()?.parse().ok()?,
                parts.next()?.parse().ok()?,
            ]),
        }),
        "set_vec3" | "set_color" | "set_color3" => Some(Command::SetParam {
            id: parts.next()?.to_string(),
            value: ParamValue::Float(vec![
                parts.next()?.parse().ok()?,
                parts.next()?.parse().ok()?,
                parts.next()?.parse().ok()?,
            ]),
        }),
        "set_vec4" => Some(Command::SetParam {
            id: parts.next()?.to_string(),
            value: ParamValue::Float(vec![
                parts.next()?.parse().ok()?,
                parts.next()?.parse().ok()?,
                parts.next()?.parse().ok()?,
                parts.next()?.parse().ok()?,
            ]),
        }),
        "set_bool" => Some(Command::SetParam {
            id: parts.next()?.to_string(),
            value: ParamValue::Bool(parts.next().map_or(true, parse_bool_text)),
        }),
        "set_i32" => Some(Command::SetParam {
            id: parts.next()?.to_string(),
            value: ParamValue::I32(parts.next()?.parse().ok()?),
        }),
        "set_u32" => Some(Command::SetParam {
            id: parts.next()?.to_string(),
            value: ParamValue::U32(parts.next()?.parse().ok()?),
        }),
        "audio" | "audio_pcm" => {
            let channels = parts.next()?.parse().ok()?;
            let interleaved = parts
                .filter_map(|value| value.parse::<f32>().ok())
                .collect::<Vec<_>>();
            if interleaved.is_empty() {
                None
            } else {
                Some(Command::PushAudio {
                    interleaved,
                    channels,
                })
            }
        }
        _ => None,
    }
}

fn parse_osc_message(message: OscMessage) -> Option<Command> {
    let addr = message.addr.trim_end_matches('/');
    let args = message.args;

    match addr {
        "/awisp/wisp/shader" => return Some(Command::LoadShader(osc_string(args.first())?)),
        "/awisp/wisp/visible" => {
            return Some(Command::SetVisible(osc_bool(args.first()).unwrap_or(true)));
        }
        "/awisp/wisp/window/titlebar" => {
            return Some(Command::SetTitleBarVisible(
                osc_bool(args.first()).unwrap_or(true),
            ));
        }
        "/awisp/wisp/window/titlebar/hide" => return Some(Command::SetTitleBarVisible(false)),
        "/awisp/wisp/window/geometry" => {
            return Some(Command::SetGeometry {
                x: osc_i32(args.first())?,
                y: osc_i32(args.get(1))?,
                width: osc_u32(args.get(2))?,
                height: osc_u32(args.get(3))?,
            });
        }
        "/awisp/wisp/image" | "/awisp/wisp/media/load" => {
            return Some(Command::LoadImage(PathBuf::from(osc_string(args.first())?)));
        }
        "/awisp/wisp/audio" | "/awisp/wisp/audio_pcm" => {
            let channels = args
                .first()
                .and_then(|arg| osc_i32(Some(arg)))
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(1)
                .max(1);
            let interleaved = args
                .iter()
                .skip(1)
                .filter_map(|arg| osc_f32(Some(arg)))
                .collect::<Vec<_>>();
            if !interleaved.is_empty() {
                return Some(Command::PushAudio {
                    interleaved,
                    channels,
                });
            }
        }
        _ => {}
    }

    let id = addr
        .strip_prefix("/awisp/wisp/param/")
        .or_else(|| addr.strip_prefix("/awisp/wisp/color/"))
        .or_else(|| addr.strip_prefix("/awisp/wisp/bool/"))
        .or_else(|| addr.strip_prefix("/awisp/wisp/checkbox/"))
        .or_else(|| addr.strip_prefix("/awisp/wisp/i32/"))
        .or_else(|| addr.strip_prefix("/awisp/wisp/u32/"))?;

    if addr
        .strip_prefix("/awisp/wisp/bool/")
        .or_else(|| addr.strip_prefix("/awisp/wisp/checkbox/"))
        .is_some()
        || args
            .first()
            .is_some_and(|arg| matches!(arg, OscType::Bool(_)))
    {
        Some(Command::SetParam {
            id: id.to_string(),
            value: ParamValue::Bool(osc_bool(args.first()).unwrap_or(true)),
        })
    } else if addr.strip_prefix("/awisp/wisp/i32/").is_some() {
        Some(Command::SetParam {
            id: id.to_string(),
            value: ParamValue::I32(osc_i32(args.first())?),
        })
    } else if addr.strip_prefix("/awisp/wisp/u32/").is_some() {
        Some(Command::SetParam {
            id: id.to_string(),
            value: ParamValue::U32(osc_u32(args.first())?),
        })
    } else {
        let values = args
            .iter()
            .filter_map(|arg| osc_f32(Some(arg)))
            .take(4)
            .collect::<Vec<_>>();
        if values.is_empty() {
            None
        } else {
            Some(Command::SetParam {
                id: id.to_string(),
                value: ParamValue::Float(values),
            })
        }
    }
}

fn osc_string(value: Option<&OscType>) -> Option<String> {
    match value? {
        OscType::String(value) => Some(value.clone()),
        _ => None,
    }
}

fn osc_f32(value: Option<&OscType>) -> Option<f32> {
    match value? {
        OscType::Float(value) => Some(*value),
        OscType::Double(value) => Some(*value as f32),
        OscType::Int(value) => Some(*value as f32),
        OscType::Long(value) => Some(*value as f32),
        _ => None,
    }
}

fn osc_i32(value: Option<&OscType>) -> Option<i32> {
    match value? {
        OscType::Int(value) => Some(*value),
        OscType::Float(value) => Some(*value as i32),
        _ => None,
    }
}

fn osc_u32(value: Option<&OscType>) -> Option<u32> {
    let value = osc_i32(value)?;
    u32::try_from(value).ok()
}

fn osc_bool(value: Option<&OscType>) -> Option<bool> {
    match value? {
        OscType::Bool(value) => Some(*value),
        OscType::Int(value) => Some(*value != 0),
        OscType::Float(value) => Some(*value != 0.0),
        _ => None,
    }
}

fn parse_text_remote_message(text: &str) -> Option<RemoteMessage> {
    let mut parts = text.split_whitespace();
    match parts.next()? {
        "discover" | "hello" => Some(RemoteMessage::Discover),
        "subscribe" => Some(RemoteMessage::Subscribe {
            enabled: parts.next().map_or(true, parse_bool_text),
        }),
        "unsubscribe" => Some(RemoteMessage::Subscribe { enabled: false }),
        _ => parse_text_command(text).map(RemoteMessage::Command),
    }
}

fn parse_osc_remote_message(bytes: &[u8]) -> Option<RemoteMessage> {
    let (_, packet) = rosc::decoder::decode_udp(bytes).ok()?;
    parse_osc_remote_packet(packet)
}

fn parse_osc_remote_packet(packet: OscPacket) -> Option<RemoteMessage> {
    match packet {
        OscPacket::Message(message) => parse_osc_remote_osc_message(message),
        OscPacket::Bundle(bundle) => bundle.content.into_iter().find_map(parse_osc_remote_packet),
    }
}

fn parse_osc_remote_osc_message(message: OscMessage) -> Option<RemoteMessage> {
    let addr = message.addr.trim_end_matches('/').to_string();
    match addr.as_str() {
        "/awisp/wisp/discover" | "/awisp/discover" => Some(RemoteMessage::Discover),
        "/awisp/wisp/subscribe" | "/awisp/subscribe" => Some(RemoteMessage::Subscribe {
            enabled: osc_bool(message.args.first()).unwrap_or(true),
        }),
        "/awisp/wisp/unsubscribe" | "/awisp/unsubscribe" => {
            Some(RemoteMessage::Subscribe { enabled: false })
        }
        _ => parse_osc_message(message).map(RemoteMessage::Command),
    }
}

fn osc_param_type_name(param_type: AwispParamType) -> &'static str {
    match param_type {
        AwispParamType::F32 => "f32",
        AwispParamType::I32 => "i32",
        AwispParamType::U32 => "u32",
        AwispParamType::Bool => "bool",
        AwispParamType::Vec2 => "vec2",
        AwispParamType::Vec3 => "vec3",
        AwispParamType::Vec4 => "vec4",
    }
}

fn osc_options_text(param: &CachedParam) -> String {
    param
        .options
        .iter()
        .map(|(value, label)| format!("{value}:{}", label.as_c_str().to_string_lossy()))
        .collect::<Vec<_>>()
        .join("|")
}

fn send_osc_to(target: SocketAddr, addr: &str, args: Vec<OscType>) {
    let packet = OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    });
    let Ok(bytes) = encoder::encode(&packet) else {
        return;
    };
    let Ok(socket) = UdpSocket::bind(("127.0.0.1", 0)) else {
        return;
    };
    let _ = socket.send_to(&bytes, target);
}

fn send_feedback_message(
    feedback_target: &Arc<Mutex<Option<SocketAddr>>>,
    feedback_enabled: &Arc<Mutex<bool>>,
    addr: &str,
    args: Vec<OscType>,
) {
    let enabled = feedback_enabled
        .lock()
        .map(|enabled| *enabled)
        .unwrap_or(false);
    if !enabled {
        return;
    }
    let target = feedback_target.lock().ok().and_then(|target| *target);
    if let Some(target) = target {
        send_osc_to(target, addr, args);
    }
}

fn send_discovery_to(target: SocketAddr, asset_root: &str, shader_name: &str) {
    send_osc_to(
        target,
        "/awisp/wisp/status",
        vec![OscType::String("ready".to_string())],
    );
    send_osc_to(
        target,
        "/awisp/wisp/shader",
        vec![OscType::String(shader_name.to_string())],
    );

    if let Err(error) = ensure_param_cache(asset_root, shader_name) {
        send_osc_to(
            target,
            "/awisp/wisp/error",
            vec![OscType::String(error.to_string())],
        );
        return;
    }

    let Ok(cache) = param_cache_cell().lock() else {
        return;
    };
    let Some(cache) = cache.as_ref() else {
        return;
    };
    send_osc_to(
        target,
        "/awisp/wisp/param_count",
        vec![OscType::Int(
            cache.params.len().min(i32::MAX as usize) as i32
        )],
    );
    for (index, param) in cache.params.iter().enumerate() {
        let id = param.id.as_c_str().to_string_lossy().into_owned();
        let label = param.label.as_c_str().to_string_lossy().into_owned();
        send_osc_to(
            target,
            "/awisp/wisp/param_desc",
            vec![
                OscType::Int(index.min(i32::MAX as usize) as i32),
                OscType::String(id),
                OscType::String(label),
                OscType::String(osc_param_type_name(param.param_type).to_string()),
                OscType::Int(param.component_count.min(i32::MAX as u32) as i32),
                OscType::Float(param.min_value),
                OscType::Float(param.max_value),
                OscType::Float(param.step),
                OscType::Float(param.default_values[0]),
                OscType::Float(param.default_values[1]),
                OscType::Float(param.default_values[2]),
                OscType::Float(param.default_values[3]),
                OscType::Bool(param.is_color),
                OscType::String(osc_options_text(param)),
            ],
        );
    }
}

fn send_discovery_from_state(
    feedback_target: &Arc<Mutex<Option<SocketAddr>>>,
    asset_root: &Arc<Mutex<String>>,
    current_shader: &Arc<Mutex<String>>,
) {
    let target = feedback_target.lock().ok().and_then(|target| *target);
    let Some(target) = target else {
        return;
    };
    let asset_root = asset_root
        .lock()
        .map(|value| value.clone())
        .unwrap_or_default();
    let shader_name = current_shader
        .lock()
        .map(|value| value.clone())
        .unwrap_or_default();
    send_discovery_to(target, &asset_root, &shader_name);
}

fn send_command_feedback(
    command: &Command,
    asset_root: &Arc<Mutex<String>>,
    current_shader: &Arc<Mutex<String>>,
    feedback_target: &Arc<Mutex<Option<SocketAddr>>>,
    feedback_enabled: &Arc<Mutex<bool>>,
) {
    match command {
        Command::LoadShader(shader_name) => {
            send_feedback_message(
                feedback_target,
                feedback_enabled,
                "/awisp/wisp/shader",
                vec![OscType::String(shader_name.clone())],
            );
            send_discovery_from_state(feedback_target, asset_root, current_shader);
        }
        Command::SetVisible(visible) => send_feedback_message(
            feedback_target,
            feedback_enabled,
            "/awisp/wisp/visible",
            vec![OscType::Bool(*visible)],
        ),
        Command::SetTitleBarVisible(visible) => send_feedback_message(
            feedback_target,
            feedback_enabled,
            "/awisp/wisp/window/titlebar",
            vec![OscType::Bool(*visible)],
        ),
        Command::SetGeometry {
            x,
            y,
            width,
            height,
        } => send_feedback_message(
            feedback_target,
            feedback_enabled,
            "/awisp/wisp/window/geometry",
            vec![
                OscType::Int(*x),
                OscType::Int(*y),
                OscType::Int((*width).min(i32::MAX as u32) as i32),
                OscType::Int((*height).min(i32::MAX as u32) as i32),
            ],
        ),
        Command::LoadImage(path) => send_feedback_message(
            feedback_target,
            feedback_enabled,
            "/awisp/wisp/image",
            vec![OscType::String(path.to_string_lossy().into_owned())],
        ),
        Command::SetParam { id, value } => {
            let args = match value {
                ParamValue::Float(values) => values.iter().copied().map(OscType::Float).collect(),
                ParamValue::I32(value) => vec![OscType::Int(*value)],
                ParamValue::U32(value) => vec![OscType::Int((*value).min(i32::MAX as u32) as i32)],
                ParamValue::Bool(value) => vec![OscType::Bool(*value)],
            };
            send_feedback_message(
                feedback_target,
                feedback_enabled,
                &format!("/awisp/wisp/param/{id}"),
                args,
            );
        }
        Command::OpenWindow { .. } | Command::PushAudio { .. } => {}
    }
}

fn send_instance_command(instance: &AwispInstance, command: Command) -> bool {
    if let Command::LoadShader(shader_name) = &command {
        if let Ok(mut current_shader) = instance.current_shader.lock() {
            *current_shader = shader_name.clone();
        }
    }
    let ok = instance
        .sender
        .send(ControlCommand {
            instance_id: instance.instance_id,
            command: command.clone(),
        })
        .is_ok();
    if ok {
        send_command_feedback(
            &command,
            &instance.asset_root,
            &instance.current_shader,
            &instance.feedback_target,
            &instance.feedback_enabled,
        );
    }
    ok
}

fn spawn_remote_listener(
    sender: Sender<ControlCommand>,
    instance_id: u64,
    port: u16,
    asset_root: Arc<Mutex<String>>,
    current_shader: Arc<Mutex<String>>,
    feedback_target: Arc<Mutex<Option<SocketAddr>>>,
    feedback_enabled: Arc<Mutex<bool>>,
) -> Result<(), String> {
    if port == 0 {
        return Ok(());
    }
    if remote_ports_cell()
        .lock()
        .map_err(|_| "awisp remote port registry poisoned".to_string())?
        .contains(&port)
    {
        return Ok(());
    }
    let socket = UdpSocket::bind(("127.0.0.1", port))
        .map_err(|error| format!("awisp udp port {port} unavailable: {error}"))?;
    remote_ports_cell()
        .lock()
        .map_err(|_| "awisp remote port registry poisoned".to_string())?
        .insert(port);
    thread::Builder::new()
        .name(format!("awisp-remote-{port}"))
        .spawn(move || {
            let mut buffer = [0_u8; 8192];
            loop {
                let Ok((len, addr)) = socket.recv_from(&mut buffer) else {
                    continue;
                };
                let message = parse_osc_remote_message(&buffer[..len]).or_else(|| {
                    std::str::from_utf8(&buffer[..len])
                        .ok()
                        .and_then(|text| parse_text_remote_message(text.trim()))
                });
                let Some(message) = message else {
                    continue;
                };
                if let Ok(mut target) = feedback_target.lock() {
                    *target = Some(addr);
                }
                match message {
                    RemoteMessage::Discover => {
                        if let Ok(mut enabled) = feedback_enabled.lock() {
                            *enabled = true;
                        }
                        let asset_root = asset_root
                            .lock()
                            .map(|value| value.clone())
                            .unwrap_or_default();
                        let shader_name = current_shader
                            .lock()
                            .map(|value| value.clone())
                            .unwrap_or_default();
                        send_discovery_to(addr, &asset_root, &shader_name);
                    }
                    RemoteMessage::Subscribe { enabled } => {
                        if let Ok(mut feedback_enabled) = feedback_enabled.lock() {
                            *feedback_enabled = enabled;
                        }
                        if enabled {
                            let asset_root = asset_root
                                .lock()
                                .map(|value| value.clone())
                                .unwrap_or_default();
                            let shader_name = current_shader
                                .lock()
                                .map(|value| value.clone())
                                .unwrap_or_default();
                            send_discovery_to(addr, &asset_root, &shader_name);
                        }
                    }
                    RemoteMessage::Command(command) => {
                        if let Command::LoadShader(shader_name) = &command {
                            if let Ok(mut current_shader) = current_shader.lock() {
                                *current_shader = shader_name.clone();
                            }
                        }
                        let _ = sender.send(ControlCommand {
                            instance_id,
                            command: command.clone(),
                        });
                        send_command_feedback(
                            &command,
                            &asset_root,
                            &current_shader,
                            &feedback_target,
                            &feedback_enabled,
                        );
                    }
                }
            }
        })
        .map_err(|error| format!("failed to spawn awisp udp listener: {error}"))?;
    Ok(())
}

fn seed_default_image_inputs(
    wisps: Res<Assets<Wisp>>,
    mut images: ResMut<Assets<Image>>,
    mut defaults: ResMut<DefaultImageInputs>,
    mut cameras: Query<(&WispHandle, &mut WispInputs)>,
) {
    for (handle, mut inputs) in &mut cameras {
        let Some(wisp) = wisps.get(&handle.0) else {
            continue;
        };
        for texture in &wisp.schema.textures {
            if !matches!(texture.role, TextureRole::ImageInput) {
                continue;
            }
            let needs_image = match inputs.get(&texture.name) {
                Some(WispValue::Image(handle)) => images.get(handle).is_none(),
                _ => true,
            };
            if !needs_image {
                continue;
            }
            let handle = defaults
                .0
                .entry(texture.name.clone())
                .or_insert_with(|| images.add(default_image_input(&texture.name)))
                .clone();
            inputs.insert(texture.name.clone(), WispValue::Image(handle));
        }
    }
}

fn poll_control(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    receiver: Res<ControlReceiver>,
    mut windows: Query<(Entity, &AwispSurface, &mut Window)>,
    mut cameras: Query<
        (
            Entity,
            &AwispSurface,
            Option<&mut WispHandle>,
            Option<&mut WispInputs>,
        ),
        With<Camera3d>,
    >,
    mut audio_feeds: ResMut<AwispAudioFeeds>,
    wisps: Res<Assets<Wisp>>,
    mut images: ResMut<Assets<Image>>,
    mut pending_params: ResMut<PendingParams>,
) {
    if !pending_params.0.is_empty() {
        let mut remaining = Vec::new();
        for (instance_id, id, value) in pending_params.0.drain(..) {
            let mut applied = false;
            let mut value = Some(value);
            for (_, surface, _, inputs) in &mut cameras {
                if surface.instance_id != instance_id {
                    continue;
                }
                if let Some(mut inputs) = inputs {
                    apply_param(&mut inputs, id.clone(), value.take().unwrap());
                    applied = true;
                    break;
                }
            }
            if !applied {
                remaining.push((instance_id, id, value.unwrap()));
            }
        }
        pending_params.0 = remaining;
    }

    let Ok(receiver) = receiver.0.lock() else {
        return;
    };
    while let Ok(control) = receiver.try_recv() {
        let instance_id = control.instance_id;
        let command = control.command;
        match command {
            Command::OpenWindow {
                shader_name,
                window_title,
                x,
                y,
                width,
                height,
            } => {
                let mut existing = false;
                for (_, surface, mut window) in &mut windows {
                    if surface.instance_id != instance_id {
                        continue;
                    }
                    window.visible = true;
                    window.title = window_title.clone();
                    window.position = WindowPosition::At(IVec2::new(x, y));
                    window
                        .resolution
                        .set_physical_resolution(width.max(1), height.max(1));
                    window.window_level = WindowLevel::AlwaysOnTop;
                    existing = true;
                }

                let wisp: Handle<Wisp> = asset_server.load(shader_name.clone());
                for (_, surface, handle, _) in &mut cameras {
                    if surface.instance_id != instance_id {
                        continue;
                    }
                    if let Some(mut handle) = handle {
                        handle.0 = wisp.clone();
                    }
                }

                if !existing {
                    let window_entity = commands
                        .spawn((
                            Window {
                                title: window_title,
                                position: WindowPosition::At(IVec2::new(x, y)),
                                resolution: WindowResolution::new(width.max(1), height.max(1)),
                                window_level: WindowLevel::AlwaysOnTop,
                                decorations: true,
                                ..default()
                            },
                            AwispSurface { instance_id },
                        ))
                        .id();
                    commands.spawn((
                        Camera3d::default(),
                        Camera::default(),
                        RenderTarget::Window(WindowRef::Entity(window_entity)),
                        WispHandle(wisp),
                        AwispSurface { instance_id },
                    ));
                }
            }
            Command::LoadShader(shader_name) => {
                let wisp: Handle<Wisp> = asset_server.load(shader_name);
                let mut updated = false;
                for (entity, surface, handle, _) in &mut cameras {
                    if surface.instance_id != instance_id {
                        continue;
                    }
                    if let Some(mut handle) = handle {
                        handle.0 = wisp.clone();
                    } else {
                        commands.entity(entity).insert(WispHandle(wisp.clone()));
                    }
                    updated = true;
                }
                if !updated {
                    let window_entity = commands
                        .spawn((
                            Window {
                                title: "Awisp".to_string(),
                                window_level: WindowLevel::AlwaysOnTop,
                                ..default()
                            },
                            AwispSurface { instance_id },
                        ))
                        .id();
                    commands.spawn((
                        Camera3d::default(),
                        Camera::default(),
                        RenderTarget::Window(WindowRef::Entity(window_entity)),
                        WispHandle(wisp),
                        AwispSurface { instance_id },
                    ));
                }
            }
            Command::SetVisible(visible) => {
                for (_, surface, mut window) in &mut windows {
                    if surface.instance_id != instance_id {
                        continue;
                    }
                    window.visible = visible;
                }
            }
            Command::SetGeometry {
                x,
                y,
                width,
                height,
            } => {
                for (_, surface, mut window) in &mut windows {
                    if surface.instance_id != instance_id {
                        continue;
                    }
                    window.position = WindowPosition::At(IVec2::new(x, y));
                    window
                        .resolution
                        .set_physical_resolution(width.max(1), height.max(1));
                    window.window_level = WindowLevel::AlwaysOnTop;
                }
            }
            Command::PushAudio {
                interleaved,
                channels,
            } => audio_feeds
                .0
                .entry(instance_id)
                .or_default()
                .push_frames(&interleaved, channels),
            Command::LoadImage(path) => {
                let image = match image_from_path(&path) {
                    Ok(image) => image,
                    Err(error) => {
                        warn!("{error}");
                        continue;
                    }
                };
                let handle = images.add(image);
                for (_, surface, handle_opt, inputs) in &mut cameras {
                    if surface.instance_id != instance_id {
                        continue;
                    }
                    let (Some(wisp_handle), Some(mut inputs)) = (handle_opt, inputs) else {
                        continue;
                    };
                    let Some(wisp) = wisps.get(&wisp_handle.0) else {
                        continue;
                    };
                    for texture in &wisp.schema.textures {
                        if matches!(texture.role, TextureRole::ImageInput) {
                            inputs.insert(texture.name.clone(), WispValue::Image(handle.clone()));
                        }
                    }
                }
            }
            Command::SetTitleBarVisible(visible) => {
                for (_, surface, mut window) in &mut windows {
                    if surface.instance_id != instance_id {
                        continue;
                    }
                    window.decorations = visible;
                }
            }
            Command::SetParam { id, value } => {
                let mut applied = false;
                let mut value = Some(value);
                for (_, surface, _, inputs) in &mut cameras {
                    if surface.instance_id != instance_id {
                        continue;
                    }
                    if let Some(mut inputs) = inputs {
                        apply_param(&mut inputs, id.clone(), value.take().unwrap());
                        applied = true;
                        break;
                    }
                }
                if !applied {
                    pending_params.0.push((instance_id, id, value.unwrap()));
                }
            }
        }
    }
}

fn update_instance_audio_textures(
    mut audio_feeds: ResMut<AwispAudioFeeds>,
    wisps: Res<Assets<Wisp>>,
    mut images: ResMut<Assets<Image>>,
    mut cameras: Query<(&AwispSurface, &WispHandle, &mut WispInputs)>,
) {
    for (surface, wisp_handle, mut inputs) in cameras.iter_mut() {
        let Some(audio) = audio_feeds.0.get_mut(&surface.instance_id) else {
            continue;
        };
        let Some(wisp) = wisps.get(&wisp_handle.0) else {
            continue;
        };
        for texture in &wisp.schema.textures {
            let (width, data) = match texture.role {
                TextureRole::AudioWaveform { samples } => {
                    (samples as usize, audio.waveform(samples as usize))
                }
                TextureRole::AudioFft { bins } => (bins as usize, audio.fft(bins as usize)),
                _ => continue,
            };
            bevy_wisp::audio::write_audio_image(
                &mut images,
                &mut inputs,
                &texture.name,
                width,
                audio.channels(),
                &data,
            );
        }
    }
}

fn hide_window_when_close_requested(
    mut close_events: MessageReader<WindowCloseRequested>,
    mut windows: Query<&mut Window>,
) {
    for event in close_events.read() {
        if let Ok(mut window) = windows.get_mut(event.window) {
            window.visible = false;
        }
    }
}

fn run_awisp_app(
    instance_id: u64,
    asset_root: String,
    shader_name: String,
    window_title: String,
    window_x: i32,
    window_y: i32,
    window_width: u32,
    window_height: u32,
    receiver: Receiver<ControlCommand>,
) {
    App::new()
        .insert_resource(StartupShader {
            instance_id,
            shader_name,
        })
        .insert_resource(WispConfig {
            audio_textures: false,
            ..default()
        })
        .insert_resource(ControlReceiver(Mutex::new(receiver)))
        .init_resource::<PendingParams>()
        .init_resource::<AwispAudioFeeds>()
        .init_resource::<DefaultImageInputs>()
        .add_plugins((
            DefaultPlugins
                .set(bevy::winit::WinitPlugin {
                    run_on_any_thread: true,
                })
                .set(AssetPlugin {
                    file_path: asset_root,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: window_title,
                        position: WindowPosition::At(IVec2::new(window_x, window_y)),
                        resolution: WindowResolution::new(window_width, window_height),
                        window_level: WindowLevel::AlwaysOnTop,
                        decorations: true,
                        ..default()
                    }),
                    exit_condition: ExitCondition::DontExit,
                    close_when_requested: false,
                    ..default()
                }),
            WispPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, seed_default_image_inputs)
        .add_systems(
            Update,
            poll_control.before(bevy_wisp::audio::update_audio_textures),
        )
        .add_systems(Update, hide_window_when_close_requested)
        .add_systems(Update, update_instance_audio_textures.after(poll_control))
        .run();
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        "awisp embedded app panicked".to_string()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn awisp_shader_count() -> usize {
    SHADERS.len()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_shader_name(index: usize) -> *const c_char {
    static NAMES: OnceLock<Vec<CString>> = OnceLock::new();
    let names = NAMES.get_or_init(|| {
        SHADERS
            .iter()
            .map(|(name, _)| CString::new(*name).unwrap())
            .collect()
    });
    names.get(index).map_or(ptr::null(), |name| name.as_ptr())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_shader_title(shader_name: *const c_char) -> *const c_char {
    static TITLES: OnceLock<Vec<CString>> = OnceLock::new();
    let Ok(shader_name) = (unsafe { opt_cstr(shader_name) }).ok_or(()) else {
        return ptr::null();
    };
    let titles = TITLES.get_or_init(|| {
        SHADERS
            .iter()
            .map(|(_, title)| CString::new(*title).unwrap())
            .collect()
    });
    SHADERS
        .iter()
        .position(|(name, _)| *name == shader_name)
        .and_then(|index| titles.get(index))
        .map_or(ptr::null(), |title| title.as_ptr())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_param_count(
    asset_root: *const c_char,
    shader_name: *const c_char,
) -> usize {
    let Some(asset_root) = (unsafe { opt_cstr(asset_root) }) else {
        set_last_error("asset_root is null");
        return 0;
    };
    let Some(shader_name) = (unsafe { opt_cstr(shader_name) }) else {
        set_last_error("shader_name is null");
        return 0;
    };
    if let Err(error) = ensure_param_cache(&asset_root, &shader_name) {
        set_last_error(error);
        return 0;
    }
    param_cache_cell()
        .lock()
        .ok()
        .and_then(|cache| cache.as_ref().map(|cache| cache.params.len()))
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_param_desc(
    asset_root: *const c_char,
    shader_name: *const c_char,
    index: usize,
    out_desc: *mut AwispParamDesc,
) -> bool {
    if out_desc.is_null() {
        set_last_error("out_desc is null");
        return false;
    }
    let Some(asset_root) = (unsafe { opt_cstr(asset_root) }) else {
        set_last_error("asset_root is null");
        return false;
    };
    let Some(shader_name) = (unsafe { opt_cstr(shader_name) }) else {
        set_last_error("shader_name is null");
        return false;
    };
    if let Err(error) = ensure_param_cache(&asset_root, &shader_name) {
        set_last_error(error);
        return false;
    }
    let Ok(cache) = param_cache_cell().lock() else {
        set_last_error("awisp param cache poisoned");
        return false;
    };
    let Some(param) = cache.as_ref().and_then(|cache| cache.params.get(index)) else {
        set_last_error("awisp param index out of range");
        return false;
    };
    unsafe {
        *out_desc = AwispParamDesc {
            id: param.id.as_ptr(),
            label: param.label.as_ptr(),
            param_type: param.param_type,
            component_count: param.component_count,
            min_value: param.min_value,
            max_value: param.max_value,
            step: param.step,
            default_values: param.default_values,
            is_color: param.is_color,
            option_count: param.options.len(),
        };
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_param_option_label(
    asset_root: *const c_char,
    shader_name: *const c_char,
    param_index: usize,
    option_index: usize,
) -> *const c_char {
    let Some(asset_root) = (unsafe { opt_cstr(asset_root) }) else {
        set_last_error("asset_root is null");
        return ptr::null();
    };
    let Some(shader_name) = (unsafe { opt_cstr(shader_name) }) else {
        set_last_error("shader_name is null");
        return ptr::null();
    };
    if let Err(error) = ensure_param_cache(&asset_root, &shader_name) {
        set_last_error(error);
        return ptr::null();
    }
    param_cache_cell()
        .lock()
        .ok()
        .and_then(|cache| {
            cache
                .as_ref()
                .and_then(|cache| cache.params.get(param_index))
                .and_then(|param| param.options.get(option_index))
                .map(|(_, label)| label.as_ptr())
        })
        .unwrap_or(ptr::null())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_param_option_value(
    asset_root: *const c_char,
    shader_name: *const c_char,
    param_index: usize,
    option_index: usize,
) -> i32 {
    let Some(asset_root) = (unsafe { opt_cstr(asset_root) }) else {
        set_last_error("asset_root is null");
        return 0;
    };
    let Some(shader_name) = (unsafe { opt_cstr(shader_name) }) else {
        set_last_error("shader_name is null");
        return 0;
    };
    if let Err(error) = ensure_param_cache(&asset_root, &shader_name) {
        set_last_error(error);
        return 0;
    }
    param_cache_cell()
        .lock()
        .ok()
        .and_then(|cache| {
            cache
                .as_ref()
                .and_then(|cache| cache.params.get(param_index))
                .and_then(|param| param.options.get(option_index))
                .map(|(value, _)| *value)
        })
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_open_embedded(
    asset_root: *const c_char,
    shader_name: *const c_char,
    window_title: *const c_char,
    window_x: i32,
    window_y: i32,
    window_width: u32,
    window_height: u32,
) -> *mut AwispInstance {
    let Some(asset_root) = (unsafe { opt_cstr(asset_root) }) else {
        set_last_error("asset_root is null");
        return ptr::null_mut();
    };
    let Some(shader_name) = (unsafe { opt_cstr(shader_name) }) else {
        set_last_error("shader_name is null");
        return ptr::null_mut();
    };
    let window_title = (unsafe { opt_cstr(window_title) }).unwrap_or_else(|| "Awisp".to_string());
    let instance_id = next_instance_id();
    if let Ok(runtime) = embedded_runtime_cell().lock() {
        if let Some(runtime) = runtime.as_ref() {
            match runtime.status.load(Ordering::Acquire) {
                STATUS_STARTING | STATUS_RUNNING => {
                    if let Ok(mut current_shader) = runtime.current_shader.lock() {
                        *current_shader = shader_name.clone();
                    }
                    let _ = runtime.sender.send(ControlCommand {
                        instance_id,
                        command: Command::OpenWindow {
                            shader_name: shader_name.clone(),
                            window_title: window_title.clone(),
                            x: window_x,
                            y: window_y,
                            width: window_width.max(1),
                            height: window_height.max(1),
                        },
                    });
                    return Box::into_raw(Box::new(AwispInstance {
                        instance_id,
                        sender: runtime.sender.clone(),
                        status: runtime.status.clone(),
                        error: runtime.error.clone(),
                        asset_root: runtime.asset_root.clone(),
                        current_shader: runtime.current_shader.clone(),
                        feedback_target: runtime.feedback_target.clone(),
                        feedback_enabled: runtime.feedback_enabled.clone(),
                    }));
                }
                STATUS_EXITED => {
                    set_last_error(
                        "awisp embedded runtime exited; restart Bespoke to open it again",
                    );
                    return ptr::null_mut();
                }
                STATUS_PANICKED => {
                    set_last_error(
                        "awisp embedded runtime panicked; restart Bespoke to open it again",
                    );
                    return ptr::null_mut();
                }
                _ => {}
            }
        }
    }

    let (sender, receiver) = mpsc::channel();
    let status = Arc::new(AtomicU8::new(STATUS_STARTING));
    let error = Arc::new(Mutex::new(CString::new("").unwrap()));
    let asset_root_state = Arc::new(Mutex::new(asset_root.clone()));
    let current_shader = Arc::new(Mutex::new(shader_name.clone()));
    let feedback_target = Arc::new(Mutex::new(None));
    let feedback_enabled = Arc::new(Mutex::new(true));
    let thread_status = status.clone();
    let thread_error = error.clone();
    let thread_asset_root = asset_root.clone();
    let thread_shader_name = shader_name.clone();
    let thread = thread::Builder::new()
        .name(format!("awisp-embedded-{shader_name}"))
        .spawn(move || {
            thread_status.store(STATUS_RUNNING, Ordering::Release);
            if let Err(error) = panic::catch_unwind(AssertUnwindSafe(|| {
                run_awisp_app(
                    instance_id,
                    thread_asset_root,
                    thread_shader_name,
                    window_title,
                    window_x,
                    window_y,
                    window_width.max(1),
                    window_height.max(1),
                    receiver,
                );
            })) {
                let message = panic_message(error);
                set_instance_error(&thread_error, message.clone());
                set_last_error(message);
                thread_status.store(STATUS_PANICKED, Ordering::Release);
            } else {
                thread_status.store(STATUS_EXITED, Ordering::Release);
            }
        });
    match thread {
        Ok(_thread) => {
            if let Ok(mut runtime) = embedded_runtime_cell().lock() {
                *runtime = Some(SharedAwispRuntime {
                    sender: sender.clone(),
                    status: status.clone(),
                    error: error.clone(),
                    asset_root: asset_root_state.clone(),
                    current_shader: current_shader.clone(),
                    feedback_target: feedback_target.clone(),
                    feedback_enabled: feedback_enabled.clone(),
                });
            }
            Box::into_raw(Box::new(AwispInstance {
                instance_id,
                sender,
                status,
                error,
                asset_root: asset_root_state,
                current_shader,
                feedback_target,
                feedback_enabled,
            }))
        }
        Err(error) => {
            set_last_error(format!("failed to spawn awisp thread: {error}"));
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_load_shader(
    instance: *mut AwispInstance,
    shader_name: *const c_char,
) -> bool {
    if instance.is_null() {
        set_last_error("awisp instance is null");
        return false;
    }
    let Some(shader_name) = (unsafe { opt_cstr(shader_name) }) else {
        set_last_error("shader_name is null");
        return false;
    };
    send_instance_command(unsafe { &*instance }, Command::LoadShader(shader_name))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_set_visible(
    instance: *mut AwispInstance,
    visible: bool,
) -> bool {
    if instance.is_null() {
        set_last_error("awisp instance is null");
        return false;
    }
    send_instance_command(unsafe { &*instance }, Command::SetVisible(visible))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_set_title_bar_visible(
    instance: *mut AwispInstance,
    visible: bool,
) -> bool {
    if instance.is_null() {
        set_last_error("awisp instance is null");
        return false;
    }
    send_instance_command(unsafe { &*instance }, Command::SetTitleBarVisible(visible))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_set_geometry(
    instance: *mut AwispInstance,
    window_x: i32,
    window_y: i32,
    window_width: u32,
    window_height: u32,
) -> bool {
    if instance.is_null() {
        set_last_error("awisp instance is null");
        return false;
    }
    send_instance_command(
        unsafe { &*instance },
        Command::SetGeometry {
            x: window_x,
            y: window_y,
            width: window_width.max(1),
            height: window_height.max(1),
        },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_push_audio(
    instance: *mut AwispInstance,
    interleaved: *const f32,
    frames: usize,
    channels: usize,
) -> bool {
    if instance.is_null() {
        set_last_error("awisp instance is null");
        return false;
    }
    if interleaved.is_null() {
        set_last_error("audio buffer is null");
        return false;
    }
    let channels = channels.max(1);
    let len = frames.saturating_mul(channels);
    let samples = unsafe { std::slice::from_raw_parts(interleaved, len) };
    send_instance_command(
        unsafe { &*instance },
        Command::PushAudio {
            interleaved: samples.to_vec(),
            channels,
        },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_load_image(
    instance: *mut AwispInstance,
    path: *const c_char,
) -> bool {
    if instance.is_null() {
        set_last_error("awisp instance is null");
        return false;
    }
    let Some(path) = (unsafe { opt_cstr(path) }) else {
        set_last_error("image path is null");
        return false;
    };
    send_instance_command(
        unsafe { &*instance },
        Command::LoadImage(PathBuf::from(path)),
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_listen_udp(
    instance: *mut AwispInstance,
    port: u16,
) -> bool {
    if instance.is_null() {
        set_last_error("awisp instance is null");
        return false;
    }
    let instance = unsafe { &*instance };
    match spawn_remote_listener(
        instance.sender.clone(),
        instance.instance_id,
        port,
        instance.asset_root.clone(),
        instance.current_shader.clone(),
        instance.feedback_target.clone(),
        instance.feedback_enabled.clone(),
    ) {
        Ok(()) => true,
        Err(error) => {
            set_last_error(error);
            false
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_set_param_f32(
    instance: *mut AwispInstance,
    id: *const c_char,
    values: *const f32,
    value_count: usize,
) -> bool {
    if instance.is_null() {
        set_last_error("awisp instance is null");
        return false;
    }
    let Some(id) = (unsafe { opt_cstr(id) }) else {
        set_last_error("param id is null");
        return false;
    };
    if values.is_null() || value_count == 0 {
        set_last_error("param values are null");
        return false;
    }
    let values = unsafe { std::slice::from_raw_parts(values, value_count.min(4)) };
    send_instance_command(
        unsafe { &*instance },
        Command::SetParam {
            id,
            value: ParamValue::Float(values.to_vec()),
        },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_set_param_bool(
    instance: *mut AwispInstance,
    id: *const c_char,
    value: bool,
) -> bool {
    if instance.is_null() {
        set_last_error("awisp instance is null");
        return false;
    }
    let Some(id) = (unsafe { opt_cstr(id) }) else {
        set_last_error("param id is null");
        return false;
    };
    send_instance_command(
        unsafe { &*instance },
        Command::SetParam {
            id,
            value: ParamValue::Bool(value),
        },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_set_param_i32(
    instance: *mut AwispInstance,
    id: *const c_char,
    value: i32,
) -> bool {
    if instance.is_null() {
        set_last_error("awisp instance is null");
        return false;
    }
    let Some(id) = (unsafe { opt_cstr(id) }) else {
        set_last_error("param id is null");
        return false;
    };
    send_instance_command(
        unsafe { &*instance },
        Command::SetParam {
            id,
            value: ParamValue::I32(value),
        },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_set_param_u32(
    instance: *mut AwispInstance,
    id: *const c_char,
    value: u32,
) -> bool {
    if instance.is_null() {
        set_last_error("awisp instance is null");
        return false;
    }
    let Some(id) = (unsafe { opt_cstr(id) }) else {
        set_last_error("param id is null");
        return false;
    };
    send_instance_command(
        unsafe { &*instance },
        Command::SetParam {
            id,
            value: ParamValue::U32(value),
        },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_free(instance: *mut AwispInstance) {
    if instance.is_null() {
        return;
    }
    let _ = unsafe { Box::from_raw(instance) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_status(instance: *const AwispInstance) -> i32 {
    if instance.is_null() {
        return STATUS_EXITED as i32;
    }
    unsafe { &*instance }.status.load(Ordering::Acquire) as i32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn awisp_instance_error(instance: *const AwispInstance) -> *const c_char {
    if instance.is_null() {
        return ptr::null();
    }
    unsafe { &*instance }
        .error
        .lock()
        .map_or(ptr::null(), |value| value.as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn awisp_last_error() -> *const c_char {
    last_error_cell()
        .lock()
        .map_or(ptr::null(), |value| value.as_ptr())
}
