export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                border: "hsl(var(--border))",
                input: "hsl(var(--input))",
                ring: "hsl(var(--ring))",
                background: "hsl(var(--background))",
                foreground: "hsl(var(--foreground))",

                // Hourglass Theme Setup
                glass: {
                    100: "rgba(255, 255, 255, 0.1)",
                    200: "rgba(255, 255, 255, 0.2)",
                    300: "rgba(255, 255, 255, 0.3)",
                    400: "rgba(255, 255, 255, 0.4)",
                    border: "rgba(255, 255, 255, 0.5)",
                },
                accent: {
                    purple: "#bcaec3",
                    dark: "#4a4a58",
                }
            },
            fontFamily: {
                sans: ["Outfit", "Inter", "sans-serif"],
            },
            borderRadius: {
                '3xl': '24px',
                '4xl': '32px',
            },
            backgroundImage: {
                'mesh-gradient': 'radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%)',
                'soft-gradient': 'linear-gradient(135deg, #e3e6ed 0%, #f7f8fa 100%)',
            }
        },
    },
    plugins: [],
}
