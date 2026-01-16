(
// Increase buffer limit
s.options.numBuffers = 1024 * 16;

// Reboot server if running with old settings
if(s.serverRunning, {
    "Rebooting server with increased buffer limit...".postln;
    s.reboot;
});

s.waitForBoot({
    var slicesDir = "/Users/vkelemen/workspace/sample-slicer/slices";
    var numGroups = 4;
    var numChannels = 2;

    // Find available sample sets
    var sampleSets = (slicesDir +/+ "*").pathMatch.select({ arg p; PathName(p).isFolder }).collect({ arg p; PathName(p).folderName });

    // Euclidean rhythm generator
    var euclidean = { arg hits, steps;
        var pattern = Array.fill(steps, 0);
        var bucket = 0;
        steps.do({ arg i;
            bucket = bucket + hits;
            if(bucket >= steps, {
                bucket = bucket - steps;
                pattern[i] = 1;
            });
        });
        pattern;
    };

    // Pattern to string
    var patternString = { arg pat;
        var chars = [". ", "x "];
        var str = "";
        pat.do({ arg x; str = str ++ chars[x] });
        str;
    };

    // Generate sample sequence for a channel
    var generateSequence = { arg ch;
        var bufs = ~channels[ch].groupBufs[~channels[ch].currentGroup];
        var numSamples = if(bufs.notNil, { bufs.size }, { 0 });
        var numHits = ~channels[ch].hits;
        ~channels[ch].sampleSeq = if(numSamples > 0, {
            Array.fill(numHits, { numSamples.rand });
        }, {
            Array.new;
        });
    };

    // Play loop for a channel (extracted to avoid duplication)
    var startPlayLoop = { arg ch;
        if(~channels[ch].running.not, {
            ~channels[ch].running = true;
            fork {
                while({ ~channels[ch].running }, {
                    var pattern = euclidean.value(~channels[ch].hits, ~channels[ch].steps);
                    var hitIdx = 0;
                    pattern.do({ arg hit;
                        if(~channels[ch].running, {
                            if(hit == 1, {
                                var bufs = ~channels[ch].groupBufs[~channels[ch].currentGroup];
                                var names = ~channels[ch].groupNames[~channels[ch].currentGroup];
                                var seq = ~channels[ch].sampleSeq;
                                if(seq.size > 0 and: { bufs.size > 0 }, {
                                    var i = seq[hitIdx % seq.size] % bufs.size;
                                    try { ~channels[ch].activeSynth.set(\gate, 0) };
                                    ~channels[ch].activeSynth = Synth(\sampler, [\buf, bufs[i], \amp, 0.5]);
                                    { ~channels[ch].label.string = names[i] }.defer;
                                });
                                hitIdx = hitIdx + 1;
                            });
                            (15 / ~tempo).wait;  // 16th note duration
                        });
                    });
                });
            };
        });
    };

    // Load samples for a given set
    var loadSampleSet = { arg ch, setName;
        var basePath = slicesDir +/+ setName;
        var oldBufs = ~channels[ch].groupBufs;

        // Stop playback and free synth
        ~channels[ch].running = false;
        try { ~channels[ch].activeSynth.set(\gate, 0) };
        try { { ~channels[ch].label.string = "--" }.defer };

        // Free old buffers
        oldBufs.do({ arg bufs; bufs.do({ arg b; b.free }) });
        s.sync;

        // Load new buffers
        ~channels[ch].groupBufs = numGroups.collect({ arg i;
            var path = basePath +/+ "group_" ++ i;
            var files = (path +/+ "*.wav").pathMatch;
            files.collect({ arg f; Buffer.read(s, f) });
        });
        ~channels[ch].groupNames = numGroups.collect({ arg i;
            var path = basePath +/+ "group_" ++ i;
            var files = (path +/+ "*.wav").pathMatch;
            files.collect({ arg f; f.basename });
        });
        s.sync;

        generateSequence.value(ch);
        ("Ch " ++ ch ++ " loaded: " ++ setName).postln;
    };

    SynthDef(\sampler, {
        arg buf = 0, amp = 0.5, gate = 1, pan = 0;
        var sig = PlayBuf.ar(2, buf, BufRateScale.kr(buf), doneAction: 0);
        var done = Done.kr(sig);
        var env = EnvGen.kr(Env.asr(0.03, 1, 0.03, \sin), gate - done, doneAction: 2);
        sig = Balance2.ar(sig[0], sig[1], pan);
        Out.ar(0, sig * amp * env);
    }).add;

    s.sync;

    // Global tempo (BPM) - each step is a 16th note
    ~tempo = 120;

    // Initialize channels
    ~channels = numChannels.collect({
        (
            groupBufs: Array.new,
            groupNames: Array.new,
            currentGroup: 0,
            activeSynth: nil,
            running: false,
            hits: 3,
            steps: 8,
            sampleSeq: Array.new
        )
    });

    // Load first sample set for each channel
    numChannels.do({ arg ch;
        loadSampleSet.value(ch, sampleSets[ch % sampleSets.size]);
    });

    s.sync;

    ~win = Window("Sample Player", Rect(100, 100, 660, 320)).front;
    ~win.background = Color.gray(0.2);

    // Build UI for each channel
    numChannels.do({ arg ch;
        var xOffset = ch * 320;
        var chColor = [Color.cyan, Color.magenta][ch];

        // Channel label
        StaticText(~win, Rect(20 + xOffset, 5, 280, 20))
            .string_("Channel " ++ (ch + 1))
            .stringColor_(chColor)
            .font_(Font("Monaco", 14, true));

        // Sample set selector
        StaticText(~win, Rect(20 + xOffset, 30, 60, 20))
            .string_("Samples:")
            .stringColor_(Color.white)
            .font_(Font("Monaco", 11));

        PopUpMenu(~win, Rect(85 + xOffset, 30, 200, 20))
            .items_(sampleSets)
            .value_(ch % sampleSets.size)
            .action_({ arg menu;
                var setName = sampleSets[menu.value];
                ~channels[ch].running = false;
                try { ~channels[ch].activeSynth.set(\gate, 0) };
                fork {
                    0.1.wait;
                    loadSampleSet.value(ch, setName);
                };
            });

        // Group selector
        StaticText(~win, Rect(20 + xOffset, 60, 50, 20))
            .string_("Group:")
            .stringColor_(Color.white)
            .font_(Font("Monaco", 11));

        ~channels[ch].groupBtns = numGroups.collect({ arg i;
            Button(~win, Rect(75 + xOffset + (i * 52), 58, 48, 24))
                .states_([
                    [i.asString, Color.white, Color.gray(0.4)],
                    [i.asString, Color.black, chColor]
                ])
                .action_({ arg btn;
                    ~channels[ch].currentGroup = i;
                    ~channels[ch].groupBtns.do({ arg b; b.value = 0 });
                    btn.value = 1;
                    generateSequence.value(ch);
                });
        });
        ~channels[ch].groupBtns[0].value = 1;

        // Euclidean controls
        StaticText(~win, Rect(20 + xOffset, 95, 40, 20))
            .string_("Hits:")
            .stringColor_(Color.white)
            .font_(Font("Monaco", 11));

        ~channels[ch].hitsBox = NumberBox(~win, Rect(60 + xOffset, 95, 40, 20))
            .value_(~channels[ch].hits)
            .decimals_(0)
            .clipLo_(1)
            .clipHi_(16);

        StaticText(~win, Rect(110 + xOffset, 95, 45, 20))
            .string_("Steps:")
            .stringColor_(Color.white)
            .font_(Font("Monaco", 11));

        ~channels[ch].stepsBox = NumberBox(~win, Rect(160 + xOffset, 95, 40, 20))
            .value_(~channels[ch].steps)
            .decimals_(0)
            .clipLo_(1)
            .clipHi_(16);

        // Pattern display
        ~channels[ch].patternLabel = StaticText(~win, Rect(20 + xOffset, 125, 220, 20))
            .string_(patternString.value(euclidean.value(~channels[ch].hits, ~channels[ch].steps)))
            .stringColor_(Color.yellow)
            .font_(Font("Monaco", 12));

        // Shuffle button
        Button(~win, Rect(245 + xOffset, 123, 50, 24))
            .states_([["SHUF", Color.white, Color.gray(0.5)]])
            .action_({
                generateSequence.value(ch);
            });

        // Update pattern
        ~channels[ch].updatePattern = {
            var pat = euclidean.value(~channels[ch].hits, ~channels[ch].steps);
            { ~channels[ch].patternLabel.string = patternString.value(pat) }.defer;
            generateSequence.value(ch);
        };

        ~channels[ch].hitsBox.action = { arg box;
            ~channels[ch].hits = box.value.asInteger;
            ~channels[ch].updatePattern.value;
        };
        ~channels[ch].stepsBox.action = { arg box;
            ~channels[ch].steps = box.value.asInteger;
            ~channels[ch].updatePattern.value;
        };

        // Sample name label
        ~channels[ch].label = StaticText(~win, Rect(20 + xOffset, 155, 280, 25));
        ~channels[ch].label.string = "--";
        ~channels[ch].label.stringColor = Color.green;
        ~channels[ch].label.font = Font("Monaco", 12);

        // Play button
        Button(~win, Rect(20 + xOffset, 190, 130, 35))
            .states_([["PLAY", Color.white, Color.green(0.6)]])
            .action_({ startPlayLoop.value(ch) });

        // Stop button
        Button(~win, Rect(160 + xOffset, 190, 130, 35))
            .states_([["STOP", Color.white, Color.red(0.6)]])
            .action_({
                ~channels[ch].running = false;
                try { ~channels[ch].activeSynth.set(\gate, 0) };
                { ~channels[ch].label.string = "--" }.defer;
            });
    });

    // Master controls
    StaticText(~win, Rect(20, 250, 60, 20))
        .string_("Master:")
        .stringColor_(Color.white)
        .font_(Font("Monaco", 12, true));

    StaticText(~win, Rect(85, 250, 35, 20))
        .string_("BPM:")
        .stringColor_(Color.white)
        .font_(Font("Monaco", 11));

    NumberBox(~win, Rect(125, 248, 50, 24))
        .value_(~tempo)
        .decimals_(0)
        .clipLo_(30)
        .clipHi_(300)
        .action_({ arg box; ~tempo = box.value });

    Button(~win, Rect(190, 248, 100, 30))
        .states_([["PLAY ALL", Color.white, Color.green(0.7)]])
        .action_({
            numChannels.do({ arg ch; startPlayLoop.value(ch) });
        });

    Button(~win, Rect(300, 248, 100, 30))
        .states_([["STOP ALL", Color.white, Color.red(0.7)]])
        .action_({
            numChannels.do({ arg ch;
                ~channels[ch].running = false;
                try { ~channels[ch].activeSynth.set(\gate, 0) };
                { ~channels[ch].label.string = "--" }.defer;
            });
        });

    ~win.onClose = {
        numChannels.do({ arg ch;
            ~channels[ch].running = false;
            ~channels[ch].groupBufs.do({ arg bufs; bufs.do({ arg b; b.free }) });
        });
    };
});
)
